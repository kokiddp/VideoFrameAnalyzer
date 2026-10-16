//! User-adjustable image filters, translated to an ffmpeg `-vf` chain.

#[derive(Clone)]
pub enum Kind {
    Float { min: f32, max: f32 },
    Choice(&'static [&'static str]),
}

#[derive(Clone)]
pub struct Param {
    pub label: &'static str,
    pub kind: Kind,
    pub value: f32,
    pub default: f32,
}

#[derive(Clone)]
pub struct Filter {
    pub id: &'static str,
    pub name: &'static str,
    pub help: &'static str,
    pub enabled: bool,
    pub params: Vec<Param>,
}

fn f(label: &'static str, min: f32, max: f32, default: f32) -> Param {
    Param { label, kind: Kind::Float { min, max }, value: default, default }
}
fn c(label: &'static str, opts: &'static [&'static str], default: usize) -> Param {
    Param { label, kind: Kind::Choice(opts), value: default as f32, default: default as f32 }
}

/// Filters in the order they are applied.
pub fn default_filters() -> Vec<Filter> {
    vec![
        Filter { id: "deint", name: "Deinterlaccia", help: "Rimuove l'interlacciamento (filmati TV/DV)",
            enabled: false, params: vec![c("Metodo", &["bwdif", "yadif"], 0)] },
        Filter { id: "denoise", name: "Riduzione rumore (veloce)", help: "hqdn3d: denoise spazio-temporale",
            enabled: false, params: vec![f("Forza", 0.0, 12.0, 4.0)] },
        Filter { id: "nlmeans", name: "Riduzione rumore (qualità, lento)", help: "Non-local means: ottima qualità, molto lento",
            enabled: false, params: vec![f("Forza", 1.0, 12.0, 4.0)] },
        Filter { id: "scale", name: "Ingrandimento Lanczos", help: "Ridimensionamento ad alta qualità (CPU)",
            enabled: false, params: vec![f("Fattore", 1.0, 4.0, 2.0)] },
        Filter { id: "placebo", name: "Ingrandimento GPU (libplacebo)", help: "Upscaler EWA/spline via Vulkan; qualsiasi GPU con Vulkan",
            enabled: false, params: vec![
                f("Fattore", 1.0, 4.0, 2.0),
                c("Algoritmo", &["ewa_lanczossharp", "ewa_lanczos", "ewa_robidouxsharp", "spline64", "bicubic"], 0),
            ] },
        Filter { id: "unsharp", name: "Nitidezza (unsharp mask)", help: "Maschera di contrasto classica",
            enabled: false, params: vec![f("Quantità", 0.0, 3.0, 1.0), c("Raggio (px)", &["3", "5", "7", "9", "11", "13"], 1)] },
        Filter { id: "cas", name: "Nitidezza adattiva (CAS)", help: "Contrast Adaptive Sharpening (AMD): poche aureole",
            enabled: false, params: vec![f("Forza", 0.0, 1.0, 0.6)] },
        Filter { id: "eq", name: "Colore / luminosità", help: "Luminosità, contrasto, saturazione, gamma",
            enabled: false, params: vec![
                f("Luminosità", -0.5, 0.5, 0.0), f("Contrasto", 0.5, 2.0, 1.0),
                f("Saturazione", 0.0, 2.0, 1.0), f("Gamma", 0.3, 3.0, 1.0),
            ] },
    ]
}

impl Filter {
    fn v(&self, i: usize) -> f32 {
        self.params[i].value
    }
    fn choice(&self, i: usize) -> &'static str {
        match self.params[i].kind {
            Kind::Choice(o) => o[(self.params[i].value as usize).min(o.len() - 1)],
            _ => "",
        }
    }

    /// ffmpeg filter expression for this filter (None when it is a no-op).
    pub fn expr(&self) -> Option<String> {
        Some(match self.id {
            "deint" => format!("{}", if self.choice(0) == "bwdif" { "bwdif=mode=send_frame" } else { "yadif=mode=send_frame" }),
            "denoise" => {
                let s = self.v(0);
                if s <= 0.0 { return None; }
                format!("hqdn3d={:.2}:{:.2}:{:.2}:{:.2}", s, s * 0.75, s * 1.5, s * 1.125)
            }
            "nlmeans" => format!("nlmeans=s={:.2}:p=7:r=15", self.v(0)),
            "scale" => {
                if self.v(0) <= 1.0 { return None; }
                format!("scale=trunc(iw*{0:.3}/2)*2:trunc(ih*{0:.3}/2)*2:flags=lanczos+accurate_rnd+full_chroma_int", self.v(0))
            }
            "placebo" => {
                if self.v(0) <= 1.0 { return None; }
                format!("libplacebo=w=trunc(iw*{0:.3}/2)*2:h=trunc(ih*{0:.3}/2)*2:upscaler={1}:downscaler=ewa_lanczos:antiringing=0.5", self.v(0), self.choice(1))
            }
            "unsharp" => {
                let sz = self.choice(1);
                let a = self.v(0);
                format!("unsharp=lx={sz}:ly={sz}:la={a:.2}:cx={sz}:cy={sz}:ca={:.2}", a * 0.5)
            }
            "cas" => format!("cas=strength={:.3}", self.v(0)),
            "eq" => {
                if (self.v(0) == 0.0) && (self.v(1) == 1.0) && (self.v(2) == 1.0) && (self.v(3) == 1.0) { return None; }
                format!("eq=brightness={:.3}:contrast={:.3}:saturation={:.3}:gamma={:.3}", self.v(0), self.v(1), self.v(2), self.v(3))
            }
            _ => return None,
        })
    }
}

/// The complete `-vf` chain for all enabled filters ("" when none).
pub fn chain(filters: &[Filter]) -> String {
    filters.iter().filter(|f| f.enabled).filter_map(|f| f.expr()).collect::<Vec<_>>().join(",")
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Engine {
    Esrgan,
    Waifu2x,
}

impl Engine {
    pub const ALL: [Engine; 2] = [Engine::Esrgan, Engine::Waifu2x];
    pub fn label(self) -> &'static str {
        match self {
            Engine::Esrgan => "Real-ESRGAN",
            Engine::Waifu2x => "waifu2x",
        }
    }
    pub fn id(self) -> &'static str {
        match self {
            Engine::Esrgan => "esrgan",
            Engine::Waifu2x => "waifu2x",
        }
    }
}

/// Neural stage settings (applied on still frames, after the ffmpeg chain).
#[derive(Clone, PartialEq)]
pub struct AiSettings {
    pub enabled: bool,
    pub engine: Engine,
    /// Real-ESRGAN model name (e.g. `realesr-animevideov3-x2`)
    pub model: String,
    pub scale: u32,
    /// waifu2x model directory suffix (`cunet`, `upconv_7_photo`, ...)
    pub w2x_model: String,
    /// waifu2x scale: 1 (denoise only, cunet), 2 or 4
    pub w2x_scale: u32,
    /// waifu2x denoise level -1 (off) .. 3
    pub w2x_noise: i32,
    /// per engine: -1 = CPU, otherwise the Vulkan GPU index
    pub dev_esrgan: i32,
    pub dev_w2x: i32,
}

impl AiSettings {
    pub fn device(&self) -> i32 {
        match self.engine {
            Engine::Esrgan => self.dev_esrgan,
            Engine::Waifu2x => self.dev_w2x,
        }
    }
    pub fn set_device(&mut self, d: i32) {
        match self.engine {
            Engine::Esrgan => self.dev_esrgan = d,
            Engine::Waifu2x => self.dev_w2x = d,
        }
    }
    /// Identity of the result an AI run produces (changes whenever the output would differ).
    pub fn key(&self) -> String {
        match self.engine {
            Engine::Esrgan => format!("esrgan:{}:{}:{}", self.model, self.scale, self.dev_esrgan),
            Engine::Waifu2x => format!("waifu2x:{}:{}:{}:{}", self.w2x_model, self.w2x_scale, self.w2x_noise, self.dev_w2x),
        }
    }
}
