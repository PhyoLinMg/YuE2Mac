//
//  SetupManager.swift — one-button setup: builds a private Python venv, downloads
//  the YuE2-3B MLX engine + chosen model variant straight from Hugging Face, and
//  verifies everything is ready. No folder picking, no external python files.
//

import Foundation
import Observation
import CryptoKit

@Observable
final class SetupManager {
    enum State { case idle, installing, ready, failed }

    /// Hugging Face repo that hosts the MLX-converted YuE2 engine. The Python
    /// modules live at the repo root; each quantized model variant is a folder
    /// (`8bit/`, `4bit/`, `bf16/`). Mirrored by `npario/YuE2-3B-MLX`.
    static let modelRepo = "ahmadw/YuE2-3B-MLX"

    /// The engine's Python modules. They run unsandboxed with the user's
    /// permissions, so they are only installed/used at the commit + SHA-256
    /// pinned in `EnginePin` (see `scripts/pin_engine.sh`).
    static let engineScripts = ["generate.py", "yue2_model.py", "yue2_vae.py"]

    var state: State = .idle
    var status = "Ready."
    var progress: Double = 0
    var errorMessage: String?

    /// Absolute path to where `generate.py` lives (its folder also holds yue2_model.py / yue2_vae.py).
    var engineRoot: String? { SettingsStore.shared.engineRoot }
    var selectedModel: String? { SettingsStore.shared.modelDir }

    init() {
        Task { await refresh() }
    }

    // MARK: Discovery

    /// Re-check everything on launch. We only ever look inside
    /// `~/Library/Application Support/YuE2Mac` — never user folders like Desktop.
    /// If the engine scripts, chosen model, or the venv are missing (e.g. the
    /// folder was deleted), drop back to the installer instead of failing later.
    func refresh() async {
        let fm = FileManager.default
        let variant = SettingsStore.shared.preferredVariant.isEmpty
            ? SystemInfo.recommendedVariant : SettingsStore.shared.preferredVariant

        let scriptsOK = Self.installedScriptsMatchPin()
        let modelOK = fm.fileExists(atPath: AppPaths.modelDir(variant).appendingPathComponent("model.safetensors").path)
        let pythonOK = fm.fileExists(atPath: AppPaths.pythonBin.path)

        await MainActor.run {
            if scriptsOK && modelOK && pythonOK {
                // Make sure the stored paths point at our always-online copies.
                SettingsStore.shared.engineRoot = AppPaths.scriptsDir.path
                SettingsStore.shared.modelDir = AppPaths.modelDir(variant).path
                state = .ready
                status = "Ready."
            } else {
                state = .idle
                status = !EnginePin.isPinned
                    ? "Engine code isn't pinned yet — run scripts/pin_engine.sh, review the files, then rebuild."
                    : settingsAreInstalled()
                    ? "The Python environment needs to be set up once."
                    : "The engine isn't installed yet — press 'Download & Install'."
            }
        }
    }

    /// True only if every engine script exists and matches its pinned SHA-256.
    /// Catches edits to the installed copies as well as an unpinned build.
    static func installedScriptsMatchPin() -> Bool {
        guard EnginePin.isPinned else { return false }
        return engineScripts.allSatisfy { name in
            guard let data = FileManager.default.contents(atPath: AppPaths.script(name).path) else { return false }
            let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            return digest == EnginePin.scriptSHA256[name]
        }
    }

    private func settingsAreInstalled() -> Bool {
        let fm = FileManager.default
        return fm.fileExists(atPath: AppPaths.modelDir(SystemInfo.recommendedVariant)
            .appendingPathComponent("model.safetensors").path)
    }

    /// Model folders already installed under the app's own `Models` directory.
    static func availableModels(engineRoot: String = "") -> [String] {
        let fm = FileManager.default
        guard let items = try? fm.contentsOfDirectory(
                at: AppPaths.modelsDir,
                includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles]) else { return [] }
        return items
            .filter(\.hasDirectoryPath)
            .map(\.path)
            .filter { fm.fileExists(atPath: URL(fileURLWithPath: $0)
                .appendingPathComponent("model.safetensors").path) }
    }

    /// A system Python to bootstrap the venv with. Prefers a 3.10–3.13 build
    /// (MLX wheels). Setups are fully automatic — never ask the user to pick.
    static func pythonCandidates() -> [String] { AppPaths.pythonCandidates }

    // MARK: Install

    func install(variant: String) async {
        let fm = FileManager.default
        state = .installing
        progress = 0
        errorMessage = nil
        do {
            try AppPaths.prepare()

            guard EnginePin.isPinned else {
                throw InstallError("Engine code isn't pinned. Run scripts/pin_engine.sh, review the downloaded Python files, then rebuild the app.")
            }

            // 1. Network reachable? We need to fetch the engine from Hugging Face.
            await setStatus("Checking internet…", 0.05)
            guard await isInternetAvailable() else {
                throw InstallError("No internet connection. YuE2Mac downloads the engine & model on first setup, then works offline.")
            }

            // 2. Pick a system Python (automatic — no user input required).
            await setStatus("Choosing Python…", 0.12)
            guard let python = Self.pythonCandidates().first else {
                throw InstallError("No compatible Python found. Install Homebrew, then run:\n  brew install python@3.12")
            }
            await setStatus("Creating the Python workspace…", 0.2)

            // 3. Create the venv.
            if !fm.fileExists(atPath: AppPaths.pythonBin.path) {
                let code = try await Shell.run(executable: python, arguments: ["-m", "venv", AppPaths.pythonDir.path]) { _ in }
                guard code == 0 else { throw InstallError("Failed to create the Python environment.") }
            }
            let pip = AppPaths.pythonDir.appendingPathComponent("bin/pip").path

            // 4. Install AI dependencies (mlx runs on Apple GPUs; huggingface_hub pulls the model).
            await setStatus("Installing MLX + friends (a minute or two)…", 0.3)
            let installCode = try await Shell.run(
                executable: pip,
                arguments: ["install", "--upgrade", "pip", "mlx", "numpy", "tiktoken", "huggingface_hub"],
                log: { _ in })
            guard installCode == 0 else { throw InstallError("Dependency install failed. Check your network and retry.") }

            // 5. Download the engine scripts + chosen model variant from Hugging Face.
            await setStatus("Downloading the \(variant) model from Hugging Face (may take a while)…", 0.5)
            try await downloadEngine(variant: variant, python: AppPaths.pythonBin.path)

            // 6. Point the app at its own copies and verify.
            SettingsStore.shared.engineRoot = AppPaths.scriptsDir.path
            SettingsStore.shared.modelDir = AppPaths.modelDir(variant).path

            await setStatus("Verifying install…", 0.92)
            let importOK = Shell.output(AppPaths.pythonBin.path, ["-c", "import mlx.core, numpy, tiktoken"]) != nil
            guard importOK else { throw InstallError("MLX won't import. See the Console log.") }

            try AppPaths.prepare() // re-ensure Output/… exist after first run
            await setStatus("Done. You’re ready to compose!", 1.0)
            state = .ready
        } catch {
            errorMessage = (error as? InstallError)?.message ?? error.localizedDescription
            state = .failed
            await setStatus(errorMessage!, 0)
        }
    }

    private func isInternetAvailable() async -> Bool {
        let url = URL(string: "https://huggingface.co")!
        var request = URLRequest(url: url)
        request.timeoutInterval = 8
        request.httpMethod = "HEAD"
        guard let (_, response) = try? await URLSession.shared.data(for: request) else { return false }
        return (response as? HTTPURLResponse)?.statusCode == 200
    }

    /// Downloads the repo's Python module files into `Scripts/` and the chosen
    /// quantized variant (`8bit/|4bit/|bf16/`) into `Models/` using huggingface_hub.
    private func downloadEngine(variant: String, python: String) async throws {
        let fm = FileManager.default
        try? fm.removeItem(at: AppPaths.stagingDir)
        try fm.createDirectory(at: AppPaths.stagingDir, withIntermediateDirectories: true)

        // Stage the exact files we need (root *.py + one variant folder), then
        // copy them into the app's self-contained layout. The `.cache` folder
        // huggingface_hub leaves behind is ignored.
        // Everything comes from the pinned commit, and the scripts are hash-checked
        // before anything is copied into place.
        let script = """
        import sys, shutil, json, hashlib
        from pathlib import Path
        from huggingface_hub import snapshot_download

        repo, variant, staging, scripts, models, revision, pins = sys.argv[1:8]
        staging, scripts, models = Path(staging), Path(scripts), Path(models)
        pins = json.loads(pins)
        local = staging / "repo"
        snapshot_download(
            repo_id=repo,
            revision=revision,
            local_dir=str(local),
            allow_patterns=[f"{variant}/*", *pins.keys()],
        )
        for name, expected in pins.items():
            digest = hashlib.sha256((local / name).read_bytes()).hexdigest()
            if digest != expected:
                print(f"HASH_MISMATCH {name} {digest}")
                sys.exit(3)
        for name in pins:
            shutil.copy2(local / name, scripts / name)
        dest = models / variant
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(local / variant, dest)
        print(f"DOWNLOAD_DONE {variant}")
        """
        let helper = AppPaths.stagingDir.appendingPathComponent("download_model.py")
        try script.write(to: helper, atomically: true, encoding: .utf8)

        let pins = String(decoding: try JSONSerialization.data(withJSONObject: EnginePin.scriptSHA256), as: UTF8.self)
        let code = try await Shell.run(
            executable: python,
            arguments: [helper.path, Self.modelRepo, variant,
                        AppPaths.stagingDir.path, AppPaths.scriptsDir.path, AppPaths.modelsDir.path,
                        EnginePin.revision, pins],
            log: { _ in })
        try? fm.removeItem(at: AppPaths.stagingDir)
        if code == 3 { throw InstallError("Downloaded engine code doesn't match the pinned SHA-256. Refusing to install it.") }
        guard code == 0 else { throw InstallError("Model download failed. Check your network and retry.") }
    }

    private struct InstallError: LocalizedError {
        let message: String
        init(_ message: String) { self.message = message }
        var errorDescription: String? { message }
    }

    private func setStatus(_ message: String, _ p: Double) async {
        await MainActor.run { status = message; progress = p }
    }
}