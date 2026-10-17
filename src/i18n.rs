//! Localization: every user-visible string goes through `tr` / `tf`.
//!
//! The text used in the source is the lookup key (Italian for the original UI strings, English for the
//! few that were already English); `TABLE` holds the Italian / English / French rendition of each key.
//! Placeholders in `tf` templates are positional: `{0}`, `{1}`, ...
use std::collections::HashMap;
use std::fmt::Display;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::OnceLock;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Lang {
    En,
    It,
    Fr,
}

impl Lang {
    pub const ALL: [Lang; 3] = [Lang::En, Lang::It, Lang::Fr];

    /// Name of the language in that language itself.
    pub fn native_name(self) -> &'static str {
        match self {
            Lang::En => "English",
            Lang::It => "Italiano",
            Lang::Fr => "Français",
        }
    }
    fn code(self) -> &'static str {
        match self {
            Lang::En => "en",
            Lang::It => "it",
            Lang::Fr => "fr",
        }
    }
    fn from_code(s: &str) -> Option<Lang> {
        let s = s.trim().to_ascii_lowercase();
        if s.starts_with("it") {
            Some(Lang::It)
        } else if s.starts_with("fr") {
            Some(Lang::Fr)
        } else if s.starts_with("en") {
            Some(Lang::En)
        } else {
            None
        }
    }
    fn index(self) -> usize {
        match self {
            Lang::It => 0,
            Lang::En => 1,
            Lang::Fr => 2,
        }
    }
}

static CURRENT: AtomicU8 = AtomicU8::new(0);

pub fn lang() -> Lang {
    match CURRENT.load(Ordering::Relaxed) {
        1 => Lang::It,
        2 => Lang::Fr,
        _ => Lang::En,
    }
}

fn set_current(l: Lang) {
    CURRENT.store(
        match l {
            Lang::En => 0,
            Lang::It => 1,
            Lang::Fr => 2,
        },
        Ordering::Relaxed,
    );
}

/// Switches language and remembers the choice for the next launch.
pub fn set_lang(l: Lang) {
    set_current(l);
    if let Some(p) = config_path() {
        if let Some(d) = p.parent() {
            let _ = std::fs::create_dir_all(d);
        }
        let _ = std::fs::write(p, l.code());
    }
}

/// Startup: saved choice, otherwise the operating system language, otherwise English.
pub fn init() {
    let saved = config_path().and_then(|p| std::fs::read_to_string(p).ok()).and_then(|s| Lang::from_code(&s));
    set_current(saved.unwrap_or_else(detect));
}

fn config_path() -> Option<PathBuf> {
    Some(dirs::config_dir()?.join("VideoFrameAnalyzer").join("language"))
}

#[cfg(windows)]
pub fn detect() -> Lang {
    #[link(name = "kernel32")]
    extern "system" {
        fn GetUserDefaultUILanguage() -> u16;
    }
    // primary language id = low 10 bits of the LANGID
    match unsafe { GetUserDefaultUILanguage() } & 0x3ff {
        0x10 => Lang::It,
        0x0c => Lang::Fr,
        _ => Lang::En,
    }
}

#[cfg(not(windows))]
pub fn detect() -> Lang {
    for var in ["LC_ALL", "LC_MESSAGES", "LANGUAGE", "LANG"] {
        if let Ok(v) = std::env::var(var) {
            // LANGUAGE may hold a list ("fr:en"); the first entry wins
            if let Some(l) = v.split(':').next().and_then(Lang::from_code) {
                return l;
            }
            if !v.is_empty() && v != "C" && v != "POSIX" {
                break;
            }
        }
    }
    Lang::En
}

type Row = (&'static str, &'static str, &'static str, &'static str);

fn index() -> &'static HashMap<&'static str, [&'static str; 3]> {
    static MAP: OnceLock<HashMap<&'static str, [&'static str; 3]>> = OnceLock::new();
    MAP.get_or_init(|| TABLE.iter().map(|r| (r.0, [r.1, r.2, r.3])).collect())
}

/// Translates a key into the current language (unknown keys are returned unchanged).
pub fn tr(key: &str) -> &str {
    match index().get(key) {
        Some(v) => v[lang().index()],
        None => key,
    }
}

/// Translates a template and fills its positional `{0}`, `{1}`, ... placeholders.
pub fn tf(key: &str, args: &[&dyn Display]) -> String {
    let mut s = tr(key).to_string();
    for (i, a) in args.iter().enumerate() {
        s = s.replace(&format!("{{{i}}}"), &a.to_string());
    }
    s
}

/// Italian-source row: (key = italian, english, french)
macro_rules! r {
    ($it:expr, $en:expr, $fr:expr) => {
        ($it, $it, $en, $fr)
    };
}
/// English-source row: (key = english, italian, french)
macro_rules! e {
    ($en:expr, $it:expr, $fr:expr) => {
        ($en, $it, $en, $fr)
    };
}

const SHORTCUTS_IT: &str = "Spazio  riproduci/pausa (o avanti 1 in modalità fotogramma)\n← →  fotogramma ±1   Maiusc ±10   Ctrl ±1 s\nHome / End  inizio / fine\nF  modalità fotogramma per fotogramma\nE o Ctrl+S  esporta\nT  OCR del fotogramma (Maiusc+trascina: area)\nI / O  ingresso / uscita CUE   Esc  deseleziona CUE\nRotella: zoom   Trascina: sposta   Doppio clic: adatta\nRotella sulle barre: zoom   Maiusc+rotella: scorri";

// (key, italian, english, french)
const TABLE: &[Row] = &[
    // ---- top bar, options menu
    r!("Apri filmato…", "Open video…", "Ouvrir une vidéo…"),
    r!("Ctrl+O, oppure trascina il file nella finestra", "Ctrl+O, or drag the file into the window", "Ctrl+O, ou glissez le fichier dans la fenêtre"),
    r!("Apri filmato", "Open video", "Ouvrir une vidéo"),
    r!("Impossibile aprire il filmato: {0}", "Cannot open the video: {0}", "Impossible d'ouvrir la vidéo : {0}"),
    r!("Opzioni", "Options", "Options"),
    r!("Lingua", "Language", "Langue"),
    r!("Informazioni", "About", "À propos"),
    r!("Sviluppato da", "Developed by", "Développé par"),
    r!("Repository del progetto", "Project repository", "Dépôt du projet"),
    r!("Versione {0}", "Version {0}", "Version {0}"),
    // ---- updates
    r!("Cerca aggiornamenti", "Check for updates", "Rechercher des mises à jour"),
    r!("Cerca aggiornamenti all'avvio", "Check for updates at startup", "Rechercher au démarrage"),
    r!("Aggiornamenti", "Updates", "Mises à jour"),
    r!("Controllo degli aggiornamenti…", "Checking for updates…", "Recherche de mises à jour…"),
    r!("Aggiornamento disponibile", "Update available", "Mise à jour disponible"),
    r!("La versione {0} è disponibile (in uso: {1}).", "Version {0} is available (installed: {1}).", "La version {0} est disponible (installée : {1})."),
    r!("Aggiorna ora", "Update now", "Mettre à jour"),
    r!("Più tardi", "Later", "Plus tard"),
    r!("Salta questa versione", "Skip this version", "Ignorer cette version"),
    r!("Aggiornamento in corso", "Updating", "Mise à jour en cours"),
    r!("Download della versione {0}…", "Downloading version {0}…", "Téléchargement de la version {0}…"),
    r!("Il programma si riavvierà al termine.", "The program will restart when done.", "Le programme redémarrera à la fin."),
    r!("Chiudi", "Close", "Fermer"),
    r!("Stai usando l'ultima versione ({0}).", "You are using the latest version ({0}).", "Vous utilisez la dernière version ({0})."),
    r!("Impossibile cercare aggiornamenti: {0}", "Cannot check for updates: {0}", "Impossible de rechercher des mises à jour : {0}"),
    r!("Aggiornamento non riuscito: {0}", "Update failed: {0}", "Échec de la mise à jour : {0}"),
    e!("curl is not available: {0}", "curl non è disponibile: {0}", "curl n'est pas disponible : {0}"),
    e!("download failed: {0}", "download non riuscito: {0}", "échec du téléchargement : {0}"),
    e!("unexpected reply from GitHub: {0}", "risposta inattesa da GitHub: {0}", "réponse inattendue de GitHub : {0}"),
    e!("unexpected reply from GitHub: no version", "risposta inattesa da GitHub: versione assente", "réponse inattendue de GitHub : version absente"),
    e!("release {0} has no build for this platform", "la release {0} non ha una versione per questa piattaforma", "la version {0} n'a pas de build pour cette plateforme"),
    e!("cannot extract the update: {0}", "impossibile estrarre l'aggiornamento: {0}", "impossible d'extraire la mise à jour : {0}"),
    e!("the downloaded file is corrupt (checksum mismatch)", "il file scaricato è corrotto (checksum errato)", "le fichier téléchargé est corrompu (somme de contrôle incorrecte)"),
    e!("the update archive does not contain the program", "l'archivio dell'aggiornamento non contiene il programma", "l'archive de mise à jour ne contient pas le programme"),
    e!("cannot replace the program ({0}): is the folder writable?", "impossibile sostituire il programma ({0}): la cartella è scrivibile?", "impossible de remplacer le programme ({0}) : le dossier est-il accessible en écriture ?"),
    // ---- transport
    r!("CUE {0} – {1}  ({2} fotogrammi, {3})", "CUE {0} – {1}  ({2} frames, {3})", "CUE {0} – {1}  ({2} images, {3})"),
    r!("Vai all'inizio", "Go to start", "Aller au début"),
    r!("Deseleziona (Esc)", "Deselect (Esc)", "Désélectionner (Échap)"),
    r!("Esporta intervallo ({0})", "Export range ({0})", "Exporter la plage ({0})"),
    r!("Inizio (Home)", "Start (Home)", "Début (Home)"),
    r!("Riproduci all'indietro", "Play backwards", "Lecture arrière"),
    r!("Fotogramma precedente (←)", "Previous frame (←)", "Image précédente (←)"),
    r!("Riproduci / pausa (Spazio)", "Play / pause (Space)", "Lecture / pause (Espace)"),
    r!("Fotogramma successivo (→)", "Next frame (→)", "Image suivante (→)"),
    r!("Fine (End)", "End (End)", "Fin (End)"),
    r!("fotogramma {0}{1}", "frame {0}{1}", "image {0}{1}"),
    r!("vai a #", "go to #", "aller à #"),
    r!("Zoom 0", "Zoom 0", "Zoom 0"),
    r!("Mostra tutta la durata nelle barre", "Show the whole duration in the bars", "Afficher toute la durée dans les barres"),
    r!("Riduci zoom barre", "Zoom the bars out", "Réduire le zoom des barres"),
    r!("Ingrandisci barre (o rotella sulle barre)", "Zoom the bars in (or mouse wheel over the bars)", "Agrandir les barres (ou molette sur les barres)"),
    r!("Velocità", "Speed", "Vitesse"),
    r!("Ripeti", "Loop", "Répéter"),
    r!("Fotogramma per fotogramma", "Frame by frame", "Image par image"),
    r!(
        "Spazio / Invio / → avanzano di un fotogramma, ← indietro, E o Ctrl+S esporta (tasto F per attivare)",
        "Space / Enter / → advance one frame, ← goes back, E or Ctrl+S exports (F key to toggle)",
        "Espace / Entrée / → avancent d'une image, ← recule, E ou Ctrl+S exporte (touche F pour activer)"
    ),
    r!(
        "Trascina (su/destra = più veloce) o usa la rotella. Doppio clic: 1×",
        "Drag (up/right = faster) or use the wheel. Double-click: 1×",
        "Glissez (haut/droite = plus rapide) ou utilisez la molette. Double-clic : 1×"
    ),
    r!(
        "CUE: trascina per selezionare un intervallo (I / O = punti di ingresso e uscita)",
        "CUE: drag to select a range (I / O = in and out points)",
        "CUE : glissez pour sélectionner une plage (I / O = points d'entrée et de sortie)"
    ),
    r!(
        "Trascina: seleziona l'intervallo · trascina i bordi gialli: regola · clic: deseleziona · rotella: zoom",
        "Drag: select the range · drag the yellow edges: adjust · click: deselect · wheel: zoom",
        "Glisser : sélectionner la plage · glisser les bords jaunes : ajuster · clic : désélectionner · molette : zoom"
    ),
    r!(
        "Rotella: zoom sul puntatore · Maiusc+rotella o trascina: scorri",
        "Wheel: zoom at the pointer · Shift+wheel or drag: scroll",
        "Molette : zoom sur le pointeur · Maj+molette ou glisser : défiler"
    ),
    // ---- filters panel
    r!("Filtri", "Filters", "Filtres"),
    r!("{0} attivi", "{0} active", "{0} actifs"),
    r!("Applica", "Apply", "Appliquer"),
    r!(
        "Disattiva per vedere l'originale. Tieni premuto C per un confronto rapido.",
        "Turn off to see the original. Hold C for a quick comparison.",
        "Désactivez pour voir l'original. Maintenez C pour une comparaison rapide."
    ),
    r!("originale (C)", "original (C)", "original (C)"),
    r!("Tutti on", "All on", "Tout activer"),
    r!("Tutti off", "All off", "Tout désactiver"),
    r!("Errore filtri: {0}", "Filter error: {0}", "Erreur de filtre : {0}"),
    r!(
        "Filtri sospesi durante la riproduzione: tornano attivi alla pausa o in modalità fotogramma per fotogramma.",
        "Filters are suspended during playback: they come back on pause or in frame-by-frame mode.",
        "Filtres suspendus pendant la lecture : ils reviennent en pause ou en mode image par image."
    ),
    r!(
        "Applicazione dei filtri in corso: attendi che il fotogramma sia aggiornato…",
        "Applying filters: wait for the frame to update…",
        "Application des filtres en cours : attendez la mise à jour de l'image…"
    ),
    r!("Reset", "Reset", "Réinitialiser"),
    r!("Ripristina il valore predefinito", "Restore the default value", "Rétablir la valeur par défaut"),
    // filter definitions (displayed through tr at draw time)
    r!("Deinterlaccia", "Deinterlace", "Désentrelacement"),
    r!("Rimuove l'interlacciamento (filmati TV/DV)", "Removes interlacing (TV/DV footage)", "Supprime l'entrelacement (vidéos TV/DV)"),
    r!("Metodo", "Method", "Méthode"),
    r!("Riduzione rumore (veloce)", "Noise reduction (fast)", "Réduction du bruit (rapide)"),
    r!("hqdn3d: denoise spazio-temporale", "hqdn3d: spatio-temporal denoise", "hqdn3d : débruitage spatio-temporel"),
    r!("Forza", "Strength", "Force"),
    r!("Riduzione rumore (qualità, lento)", "Noise reduction (quality, slow)", "Réduction du bruit (qualité, lent)"),
    r!("Non-local means: ottima qualità, molto lento", "Non-local means: excellent quality, very slow", "Non-local means : excellente qualité, très lent"),
    r!("Ingrandimento Lanczos", "Lanczos upscale", "Agrandissement Lanczos"),
    r!("Ridimensionamento ad alta qualità (CPU)", "High-quality resizing (CPU)", "Redimensionnement de haute qualité (CPU)"),
    r!("Fattore", "Factor", "Facteur"),
    r!("Ingrandimento GPU (libplacebo)", "GPU upscale (libplacebo)", "Agrandissement GPU (libplacebo)"),
    r!(
        "Upscaler EWA/spline via Vulkan; qualsiasi GPU con Vulkan",
        "EWA/spline upscaler via Vulkan; any Vulkan-capable GPU",
        "Upscaler EWA/spline via Vulkan ; tout GPU compatible Vulkan"
    ),
    r!("Algoritmo", "Algorithm", "Algorithme"),
    r!("Nitidezza (unsharp mask)", "Sharpening (unsharp mask)", "Netteté (masque flou)"),
    r!("Maschera di contrasto classica", "Classic unsharp mask", "Masque de netteté classique"),
    r!("Quantità", "Amount", "Quantité"),
    r!("Raggio (px)", "Radius (px)", "Rayon (px)"),
    r!("Nitidezza adattiva (CAS)", "Adaptive sharpening (CAS)", "Netteté adaptative (CAS)"),
    r!("Contrast Adaptive Sharpening (AMD): poche aureole", "Contrast Adaptive Sharpening (AMD): few halos", "Contrast Adaptive Sharpening (AMD) : peu de halos"),
    r!("Colore / luminosità", "Color / brightness", "Couleur / luminosité"),
    r!("Luminosità, contrasto, saturazione, gamma", "Brightness, contrast, saturation, gamma", "Luminosité, contraste, saturation, gamma"),
    r!("Luminosità", "Brightness", "Luminosité"),
    r!("Contrasto", "Contrast", "Contraste"),
    r!("Saturazione", "Saturation", "Saturation"),
    r!("Gamma", "Gamma", "Gamma"),
    // ---- neural network card
    r!("elaborazione… {0}s", "processing… {0}s", "traitement… {0} s"),
    r!("errore: {0}", "error: {0}", "erreur : {0}"),
    r!("in pausa durante la riproduzione", "paused during playback", "en pause pendant la lecture"),
    r!("applicata", "applied", "appliquée"),
    r!("Rete neurale", "Neural network", "Réseau de neurones"),
    r!(
        "Super-risoluzione / pulizia con rete neurale (GPU Vulkan o CPU).\nIn anteprima viene applicata solo al fotogramma fermo.",
        "Super-resolution / cleanup with a neural network (Vulkan GPU or CPU).\nIn the preview it is only applied to a paused frame.",
        "Super-résolution / nettoyage par réseau de neurones (GPU Vulkan ou CPU).\nDans l'aperçu, il n'est appliqué qu'à l'image figée."
    ),
    r!("Motore", "Engine", "Moteur"),
    r!("Processore", "Processor", "Processeur"),
    r!("test in corso…", "testing…", "test en cours…"),
    r!("nessuno disponibile", "none available", "aucun disponible"),
    r!(" · più veloce", " · fastest", " · le plus rapide"),
    r!("Test", "Test", "Test"),
    r!("Ripeti il test prestazioni (entrambi i motori)", "Repeat the performance test (both engines)", "Refaire le test de performances (les deux moteurs)"),
    r!("Modello", "Model", "Modèle"),
    r!("fisso", "fixed", "fixe"),
    r!(
        "animevideov3 = veloce · x4plus = massima qualità",
        "animevideov3 = fast · x4plus = best quality",
        "animevideov3 = rapide · x4plus = qualité maximale"
    ),
    r!("foto (veloce)", "photo (fast)", "photo (rapide)"),
    r!("anime/disegni (veloce)", "anime/drawings (fast)", "anime/dessins (rapide)"),
    r!("cunet (qualità, più lento)", "cunet (quality, slower)", "cunet (qualité, plus lent)"),
    r!("Pulizia", "Cleanup", "Nettoyage"),
    r!("nessuna", "none", "aucun"),
    r!("livello {0}", "level {0}", "niveau {0}"),
    r!("1× (solo pulizia)", "1× (cleanup only)", "1× (nettoyage seul)"),
    r!(
        "waifu2x: più leggero di Real-ESRGAN; la pulizia riduce rumore/blocchi di compressione",
        "waifu2x: lighter than Real-ESRGAN; cleanup reduces noise and compression blocks",
        "waifu2x : plus léger que Real-ESRGAN ; le nettoyage réduit le bruit et les blocs de compression"
    ),
    r!("Nessun motore neurale incluso in questa build.", "No neural engine is included in this build.", "Aucun moteur neuronal n'est inclus dans cette version."),
    r!("Benchmark…", "Benchmark…", "Benchmark…"),
    r!(
        "Confronta modelli, risoluzioni e processori (CPU/GPU)",
        "Compare models, resolutions and processors (CPU/GPU)",
        "Comparer modèles, résolutions et processeurs (CPU/GPU)"
    ),
    // ---- export section
    r!("Esporta fotogrammi", "Export frames", "Exporter les images"),
    r!(
        "Si esporta con {0} filtri/elaborazioni attivi (come l'anteprima)",
        "Exporting with {0} active filters/processing steps (same as the preview)",
        "Export avec {0} filtres/traitements actifs (comme l'aperçu)"
    ),
    r!(
        "«Applica» è attivo ma nessun filtro è selezionato: fotogrammi originali",
        "\"Apply\" is on but no filter is selected: original frames",
        "« Appliquer » est activé mais aucun filtre n'est sélectionné : images originales"
    ),
    r!(
        "«Applica» è disattivato: si esportano i fotogrammi originali",
        "\"Apply\" is off: the original frames are exported",
        "« Appliquer » est désactivé : les images originales sont exportées"
    ),
    r!("Fotogrammi consecutivi", "Consecutive frames", "Images consécutives"),
    r!("Cartella di destinazione…", "Destination folder…", "Dossier de destination…"),
    r!("Cartella di esportazione", "Export folder", "Dossier d'exportation"),
    r!("Esporta fotogramma corrente", "Export current frame", "Exporter l'image courante"),
    r!("E oppure Ctrl+S", "E or Ctrl+S", "E ou Ctrl+S"),
    r!("Mostra nella cartella", "Show in folder", "Afficher dans le dossier"),
    r!("Scorciatoie", "Shortcuts", "Raccourcis"),
    (
        SHORTCUTS_IT,
        SHORTCUTS_IT,
        "Space  play/pause (or forward 1 in frame-by-frame mode)\n← →  frame ±1   Shift ±10   Ctrl ±1 s\nHome / End  start / end\nF  frame-by-frame mode\nE or Ctrl+S  export\nT  OCR of the frame (Shift+drag: area)\nI / O  CUE in / out   Esc  deselect CUE\nWheel: zoom   Drag: pan   Double-click: fit\nWheel over the bars: zoom   Shift+wheel: scroll",
        "Espace  lecture/pause (ou avance de 1 en mode image par image)\n← →  image ±1   Maj ±10   Ctrl ±1 s\nHome / End  début / fin\nF  mode image par image\nE ou Ctrl+S  exporter\nT  OCR de l'image (Maj+glisser : zone)\nI / O  entrée / sortie CUE   Échap  désélectionner CUE\nMolette : zoom   Glisser : déplacer   Double-clic : ajuster\nMolette sur les barres : zoom   Maj+molette : défiler"
    ),
    r!("Pixel esatti (zoom senza smussare)", "Exact pixels (zoom without smoothing)", "Pixels exacts (zoom sans lissage)"),
    e!("PNG (lossless)", "PNG (senza perdita)", "PNG (sans perte)"),
    e!("TIFF (lossless)", "TIFF (senza perdita)", "TIFF (sans perte)"),
    r!("JPEG (qualità massima)", "JPEG (maximum quality)", "JPEG (qualité maximale)"),
    e!("WebP (lossless)", "WebP (senza perdita)", "WebP (sans perte)"),
    r!("BMP (non compresso)", "BMP (uncompressed)", "BMP (non compressé)"),
    r!("Avvio…", "Starting…", "Démarrage…"),
    r!("Esportati {0} file. Ultimo: {1}", "Exported {0} files. Last: {1}", "{0} fichiers exportés. Dernier : {1}"),
    r!("Annullato", "Cancelled", "Annulé"),
    r!("Export fallito:\n{0}", "Export failed:\n{0}", "Échec de l'exportation :\n{0}"),
    r!("cartella: {0}", "folder: {0}", "dossier : {0}"),
    r!("Fotogramma {0}/{1}", "Frame {0}/{1}", "Image {0}/{1}"),
    r!("Estrazione {0}/{1}", "Extracting {0}/{1}", "Extraction {0}/{1}"),
    r!("Rete neurale: fotogramma {0}/{1}", "Neural network: frame {0}/{1}", "Réseau de neurones : image {0}/{1}"),
    r!(
        "Nessun fotogramma esportato (posizione oltre la fine del filmato?)",
        "No frame exported (position past the end of the video?)",
        "Aucune image exportée (position après la fin de la vidéo ?)"
    ),
    r!("Completato", "Done", "Terminé"),
    r!("Esportazione in corso", "Export in progress", "Exportation en cours"),
    r!("Trascorso {0}", "Elapsed {0}", "Écoulé {0}"),
    r!("Rimanente: pochi istanti", "Remaining: a few moments", "Restant : quelques instants"),
    r!("Rimanente ~{0}", "Remaining ~{0}", "Restant ~{0}"),
    r!("Rimanente: calcolo…", "Remaining: calculating…", "Restant : calcul…"),
    r!("Annulla", "Cancel", "Annuler"),
    // ---- benchmark window
    r!("Benchmark rete neurale", "Neural network benchmark", "Benchmark du réseau de neurones"),
    r!("Modelli", "Models", "Modèles"),
    r!("Risoluzione", "Resolution", "Résolution"),
    r!("Processori", "Processors", "Processeurs"),
    r!("Tempo massimo per prova", "Maximum time per test", "Durée maximale par essai"),
    r!(
        "Oltre il limite la prova è interrotta e le risoluzioni maggiori dello stesso modello sono saltate.",
        "Beyond the limit the test is aborted and higher resolutions of the same model are skipped.",
        "Au-delà de la limite, l'essai est interrompu et les résolutions supérieures du même modèle sont ignorées."
    ),
    r!("rimanente ~{0}", "remaining ~{0}", "restant ~{0}"),
    r!("rimanente: calcolo…", "remaining: calculating…", "restant : calcul…"),
    r!("Avvia benchmark", "Start benchmark", "Lancer le benchmark"),
    r!("attendi la fine del test iniziale…", "wait for the initial test to finish…", "attendez la fin du test initial…"),
    r!(
        "seleziona almeno un modello, una risoluzione e un processore",
        "select at least one model, one resolution and one processor",
        "sélectionnez au moins un modèle, une résolution et un processeur"
    ),
    r!("Copia (Markdown)", "Copy (Markdown)", "Copier (Markdown)"),
    r!("Copia (CSV)", "Copy (CSV)", "Copier (CSV)"),
    r!(
        "Immagine di prova: il fotogramma corrente, ridimensionato a ogni risoluzione.",
        "Test image: the current frame, resized to each resolution.",
        "Image de test : l'image courante, redimensionnée à chaque résolution."
    ),
    r!(
        "Immagine di prova: sintetica (apri un filmato per usare un suo fotogramma).",
        "Test image: synthetic (open a video to use one of its frames).",
        "Image de test : synthétique (ouvrez une vidéo pour utiliser l'une de ses images)."
    ),
    r!(
        "Misura il tempo per fotogramma di ogni modello, a diverse risoluzioni, su CPU e GPU.",
        "Measures the time per frame of each model, at different resolutions, on CPU and GPU.",
        "Mesure le temps par image de chaque modèle, à différentes résolutions, sur CPU et GPU."
    ),
    r!("saltato", "skipped", "ignoré"),
    r!("CPU ({0} thread)", "CPU ({0} threads)", "CPU ({0} threads)"),
    // ---- log panel
    r!("Log", "Log", "Journal"),
    r!("Mostra / nascondi (tasto L)", "Show / hide (L key)", "Afficher / masquer (touche L)"),
    r!("Copia", "Copy", "Copier"),
    r!("Pulisci", "Clear", "Effacer"),
    r!("{0} righe", "{0} lines", "{0} lignes"),
    // ---- viewer
    r!("Errore: {0}", "Error: {0}", "Erreur : {0}"),
    r!(
        "Preparazione componenti (solo al primo avvio)…",
        "Preparing components (first launch only)…",
        "Préparation des composants (premier lancement uniquement)…"
    ),
    r!("Trascina qui un filmato", "Drop a video here", "Déposez une vidéo ici"),
    r!("oppure premi Ctrl+O", "or press Ctrl+O", "ou appuyez sur Ctrl+O"),
    r!("Caricamento…", "Loading…", "Chargement…"),
    r!("Rete neurale… {0}s", "Neural network… {0}s", "Réseau de neurones… {0} s"),
    r!("Applicazione filtri…", "Applying filters…", "Application des filtres…"),
    r!("Rete neurale…", "Neural network…", "Réseau de neurones…"),
    r!(
        "Rete neurale attiva solo a fotogramma fermo",
        "Neural network only active on a paused frame",
        "Réseau de neurones actif uniquement sur une image figée"
    ),
    r!("AI: {0}", "AI: {0}", "IA : {0}"),
    r!("Adatta", "Fit", "Ajuster"),
    r!("Doppio clic sull'immagine", "Double-click the image", "Double-clic sur l'image"),
    r!("Immagine {0}×{1} oltre il limite della GPU ({2})", "Image {0}×{1} exceeds the GPU limit ({2})", "Image {0}×{1} au-delà de la limite du GPU ({2})"),
    r!(
        "Risultato {0}x{1} oltre il limite della GPU ({2}); usa un fattore minore",
        "Result {0}x{1} exceeds the GPU limit ({2}); use a smaller factor",
        "Résultat {0}x{1} au-delà de la limite du GPU ({2}) ; utilisez un facteur plus petit"
    ),
    // ---- neural network / engines errors
    r!("{0} non incluso in questo eseguibile", "{0} is not included in this executable", "{0} n'est pas inclus dans cet exécutable"),
    r!("{0} non avviabile: {1}", "{0} cannot be started: {1}", "{0} ne peut pas démarrer : {1}"),
    r!("{0} fallito ({1}); {2}\n{3}", "{0} failed ({1}); {2}\n{3}", "{0} a échoué ({1}) ; {2}\n{3}"),
    r!("GPU Vulkan disponibile?", "Is a Vulkan GPU available?", "Un GPU Vulkan est-il disponible ?"),
    r!("frame non valido", "invalid frame", "image invalide"),
    r!("nessuna esecuzione", "no run", "aucune exécution"),
    // ---- OCR
    r!("area OCR", "OCR area", "zone OCR"),
    r!("OCR", "OCR", "OCR"),
    r!("Copiato: {0}", "Copied: {0}", "Copié : {0}"),
    r!("Nessun intervallo CUE selezionato (tasti I / O)", "No CUE range selected (I / O keys)", "Aucune plage CUE sélectionnée (touches I / O)"),
    r!("{0} risultati ({1} targhe) in {2}s", "{0} results ({1} plates) in {2}s", "{0} résultats ({1} plaques) en {2} s"),
    r!(
        "tipo,testo,confidenza,formato,frame_inizio,frame_fine,frame_migliore,tempo_s,letture,alternative\n",
        "type,text,confidence,format,start_frame,end_frame,best_frame,time_s,readings,alternatives\n",
        "type,texte,confiance,format,image_début,image_fin,meilleure_image,temps_s,lectures,alternatives\n"
    ),
    r!("targa", "plate", "plaque"),
    r!("testo", "text", "texte"),
    r!("TARGA", "PLATE", "PLAQUE"),
    r!("Modelli OCR non inclusi in questa build.", "OCR models are not included in this build.", "Les modèles OCR ne sont pas inclus dans cette version."),
    r!("Testo", "Text", "Texte"),
    r!("Targhe", "Plates", "Plaques"),
    r!("Entrambi", "Both", "Les deux"),
    r!("Fotogramma", "Frame", "Image"),
    r!("Intervallo CUE", "CUE range", "Plage CUE"),
    r!("Serie", "Series", "Série"),
    r!("Fotogrammi:", "Frames:", "Images :"),
    r!("Analizza uno ogni:", "Analyze one every:", "Analyser une sur :"),
    r!(
        "Più fotogrammi = voto più affidabile, ma più lento (circa 1 s per fotogramma Full HD su CPU).",
        "More frames = more reliable voting, but slower (about 1 s per Full HD frame on CPU).",
        "Plus d'images = vote plus fiable, mais plus lent (environ 1 s par image Full HD sur CPU)."
    ),
    r!("Usa fotogramma potenziato (AI)", "Use enhanced frame (AI)", "Utiliser l'image améliorée (IA)"),
    r!(
        "Passa ogni fotogramma dalla rete neurale prima della lettura: aiuta su scritte piccole e sfocate, ma è molto più lento. Richiede l'AI attiva nei filtri.",
        "Runs each frame through the neural network before reading: helps with small, blurry text, but is much slower. Requires AI to be enabled in the filters.",
        "Fait passer chaque image par le réseau de neurones avant la lecture : aide pour les textes petits et flous, mais bien plus lent. Nécessite que l'IA soit activée dans les filtres."
    ),
    r!("Usa la GPU (DirectML)", "Use the GPU (DirectML)", "Utiliser le GPU (DirectML)"),
    r!(
        "Esegue i modelli sulla scheda video (NVIDIA, AMD o Intel con DirectX 12): molto più veloce. Se non parte si usa la CPU.",
        "Runs the models on the graphics card (NVIDIA, AMD or Intel with DirectX 12): much faster. If it fails to start, the CPU is used.",
        "Exécute les modèles sur la carte graphique (NVIDIA, AMD ou Intel avec DirectX 12) : beaucoup plus rapide. En cas d'échec, le CPU est utilisé."
    ),
    r!("OCR automatico a fotogramma fermo", "Automatic OCR on a paused frame", "OCR automatique sur image figée"),
    r!("Area: selezionata", "Area: selected", "Zone : sélectionnée"),
    r!("Tutto il fotogramma", "Whole frame", "Image entière"),
    r!("Area: tutto il fotogramma", "Area: whole frame", "Zone : image entière"),
    r!("(Maiusc+trascina sull'immagine per limitarla)", "(Shift+drag on the image to limit it)", "(Maj+glisser sur l'image pour la limiter)"),
    r!("Leggi (T)", "Read (T)", "Lire (T)"),
    r!("Confidenza minima", "Minimum confidence", "Confiance minimale"),
    r!("Solo targhe con formato valido", "Only plates with a valid format", "Uniquement les plaques au format valide"),
    r!("Formato: {0}", "Format: {0}", "Format : {0}"),
    r!(
        "{0} letture, fotogrammi {1}–{2} (migliore: {3})",
        "{0} readings, frames {1}–{2} (best: {3})",
        "{0} lectures, images {1}–{2} (meilleure : {3})"
    ),
    r!("Altre letture: {0}", "Other readings: {0}", "Autres lectures : {0}"),
    r!("Clic: vai al fotogramma · clic destro: copia", "Click: go to frame · right-click: copy", "Clic : aller à l'image · clic droit : copier"),
    r!("Copia tutto", "Copy all", "Tout copier"),
    r!("Copiati {0} risultati", "Copied {0} results", "{0} résultats copiés"),
    r!("Salva CSV", "Save CSV", "Enregistrer le CSV"),
    r!("Salvato {0}", "Saved {0}", "Enregistré : {0}"),
    r!("CSV non salvato: {0}", "CSV not saved: {0}", "CSV non enregistré : {0}"),
    r!("Scegli prima la cartella di esportazione", "Choose the export folder first", "Choisissez d'abord le dossier d'exportation"),
    r!("IT storica", "IT historic", "IT historique"),
    r!("FR vecchia", "FR old", "FR ancienne"),
    r!("UK vecchia", "UK old", "UK ancien"),
    r!("modelli OCR non inclusi in questa build", "OCR models are not included in this build", "modèles OCR non inclus dans cette version"),
    r!("nessun fotogramma decodificato", "no frame decoded", "aucune image décodée"),
    r!("modelli OCR mancanti in {0}", "OCR models missing in {0}", "modèles OCR manquants dans {0}"),
    r!(
        "uscita del detector inattesa ({0} valori per {1}x{2})",
        "unexpected detector output ({0} values for {1}x{2})",
        "sortie inattendue du détecteur ({0} valeurs pour {1}x{2})"
    ),
    r!("uscita OCR targhe inattesa ({0} valori)", "unexpected plate OCR output ({0} values)", "sortie inattendue de l'OCR des plaques ({0} valeurs)"),
    // ---- start-up / ffmpeg errors (English in the source)
    e!("cannot run ffmpeg: {0}", "impossibile eseguire ffmpeg: {0}", "impossible d'exécuter ffmpeg : {0}"),
    e!("no video stream found ({0})", "nessun flusso video trovato ({0})", "aucun flux vidéo trouvé ({0})"),
    e!("cannot determine video size", "impossibile determinare le dimensioni del video", "impossible de déterminer la taille de la vidéo"),
    e!("cannot create {0}: {1}", "impossibile creare {0}: {1}", "impossible de créer {0} : {1}"),
    e!("unpack failed: {0}", "estrazione non riuscita: {0}", "échec de l'extraction : {0}"),
    e!("ffmpeg missing from the embedded bundle", "ffmpeg assente dal pacchetto incorporato", "ffmpeg absent du paquet intégré"),
    e!("cannot start ffmpeg: {0}", "impossibile avviare ffmpeg: {0}", "impossible de lancer ffmpeg : {0}"),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_is_consistent() {
        let mut seen = std::collections::HashSet::new();
        for (key, it, en, fr) in TABLE {
            assert!(seen.insert(*key), "duplicate key {key:?}");
            for t in [it, en, fr] {
                assert!(!t.is_empty(), "empty translation for {key:?}");
            }
            let n = |s: &str| (0..10).filter(|i| s.contains(&format!("{{{i}}}"))).count();
            assert!(n(it) == n(en) && n(en) == n(fr), "placeholder mismatch for {key:?}");
        }
    }

    /// Every literal passed to `tr("..")` / `tf("..")` in the sources must have a table row.
    #[test]
    fn all_call_sites_are_translated() {
        fn unescape(s: &str) -> String {
            s.replace("\\n", "\n").replace("\\\"", "\"").replace("\\'", "'").replace("\\\\", "\\")
        }
        fn walk(dir: &std::path::Path, out: &mut Vec<PathBuf>) {
            for e in std::fs::read_dir(dir).unwrap().flatten() {
                let p = e.path();
                if p.is_dir() {
                    walk(&p, out);
                } else if p.extension().is_some_and(|x| x == "rs") {
                    out.push(p);
                }
            }
        }
        let mut files = Vec::new();
        walk(&PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src"), &mut files);
        let keys: std::collections::HashSet<&str> = TABLE.iter().map(|r| r.0).collect();
        let mut missing = Vec::new();
        for f in files {
            if f.file_name().is_some_and(|n| n == "i18n.rs" || n == "selftest.rs") {
                continue;
            }
            let src = std::fs::read_to_string(&f).unwrap();
            for pat in ["tr(\"", "tf(\""] {
                let mut rest = src.as_str();
                while let Some(i) = rest.find(pat) {
                    // skip identifiers that merely end in tr/tf (e.g. `str("`)
                    let before = rest[..i].chars().last();
                    let body = &rest[i + pat.len()..];
                    let mut end = 0;
                    let b = body.as_bytes();
                    while end < b.len() && b[end] != b'"' {
                        end += if b[end] == b'\\' { 2 } else { 1 };
                    }
                    let end = end.min(b.len());
                    if !before.is_some_and(|c| c.is_alphanumeric() || c == '_') {
                        let key = unescape(&body[..end]);
                        if !keys.contains(key.as_str()) {
                            missing.push(format!("{}: {key:?}", f.display()));
                        }
                    }
                    rest = &body[end..];
                }
            }
        }
        assert!(missing.is_empty(), "strings without translation:\n{}", missing.join("\n"));
    }

    #[test]
    fn filter_texts_are_translated() {
        let keys: std::collections::HashSet<&str> = TABLE.iter().map(|r| r.0).collect();
        for f in crate::filters::default_filters() {
            assert!(keys.contains(f.name), "{}", f.name);
            assert!(keys.contains(f.help), "{}", f.help);
            for p in &f.params {
                assert!(keys.contains(p.label), "{}", p.label);
            }
        }
    }

    #[test]
    fn lookup_and_format() {
        set_current(Lang::Fr);
        assert_eq!(tf("{0} attivi", &[&3]), "3 actifs");
        set_current(Lang::En);
        assert_eq!(tr("Annulla"), "Cancel");
        set_current(Lang::It);
        assert_eq!(tr("Annulla"), "Annulla");
        assert_eq!(tr("not a key"), "not a key");
    }
}
