use cushy::{
    context::GraphicsContext,
    kludgine::{
        app::winit::keyboard::{KeyCode, ModifiersKeyState},
        figures::{units::Px, FloatConversion, Point, Px2D, Rect, Size},
        shapes::{Path, PathBuilder, Shape, StrokeOptions},
        text::Text,
        Color, DrawableExt,
    },
    widgets::Canvas,
    Run, Tick,
};
use rand::Rng;

const ELECTRON_COUNT: usize = 2;
const POSITRON_COUNT: usize = 2;
const SIMULATION_SIZE: usize = 250;
const MARGIN: f32 = 50.0;
const INIT_VEL: f64 = 1.2;

const COUPLING: f64 = 10.0;
const DISSIPATION: f64 = 0.6;
const DIFFUSION: f64 = 0.6;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct GuardedBool {
    on: bool,
    guard: bool,
}
impl From<bool> for GuardedBool {
    fn from(value: bool) -> Self {
        Self {
            on: value,
            guard: false,
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct TimedBool {
    on: bool,
    elapsed: usize,
}
impl From<bool> for TimedBool {
    fn from(value: bool) -> Self {
        Self {
            on: value,
            elapsed: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum ExprTok {
    Num(f64),
    R,
    A,
    B,
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct Particle {
    pos: [f64; 2],
    vel: [f64; 2],
    charge: f64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct Excitation {
    abs: f64,
    change: f64,
}

struct ParticleSystem {
    electrons: [Particle; ELECTRON_COUNT],
    positrons: [Particle; POSITRON_COUNT],
    photons: [[Excitation; SIMULATION_SIZE]; SIMULATION_SIZE],
    dt: f64,

    keyboard: [(char, GuardedBool, KeyCode); 38],
    plus_guard: GuardedBool,
    times_guard: GuardedBool,
    pow_guard: GuardedBool,
    backspace_guard: GuardedBool,
    arr_l_guard: GuardedBool,
    arr_r_guard: GuardedBool,
    dot_guard: GuardedBool,
    space_guard: GuardedBool,
    force_def_str1: String, // Before cursor
    force_def_str2: String, // After cursor
    force_prog: Vec<ExprTok>,
    invalid_prog: bool,
    comp_stack: Vec<f64>,
}

// Particle behaviour
//
impl ParticleSystem {
    fn new() -> Self {
        let mut rng = rand::thread_rng();
        let mut electrons: [Particle; ELECTRON_COUNT] =
            [Default::default(); ELECTRON_COUNT];
        for e in &mut electrons {
            e.charge = -1.0;
            e.pos = [
                rng.gen_range(0.0..SIMULATION_SIZE as f64),
                rng.gen_range(0.0..SIMULATION_SIZE as f64),
            ];
            e.vel = [
                rng.gen_range(-INIT_VEL..INIT_VEL),
                rng.gen_range(-INIT_VEL..INIT_VEL),
            ];
        }

        let mut positrons: [Particle; POSITRON_COUNT] =
            [Default::default(); POSITRON_COUNT];
        for p in &mut positrons {
            p.charge = 1.0;
            p.pos = [
                rng.gen_range(0.0..SIMULATION_SIZE as f64),
                rng.gen_range(0.0..SIMULATION_SIZE as f64),
            ];
            p.vel = [
                rng.gen_range(-INIT_VEL..INIT_VEL),
                rng.gen_range(-INIT_VEL..INIT_VEL),
            ];
        }

        let photons: [[Excitation; SIMULATION_SIZE]; SIMULATION_SIZE] =
            [[Default::default(); SIMULATION_SIZE]; SIMULATION_SIZE];

        Self {
            electrons,
            positrons,
            photons,
            dt: 0.3,

            keyboard: [
                ('a', Default::default(), KeyCode::KeyA),
                ('b', Default::default(), KeyCode::KeyB),
                ('c', Default::default(), KeyCode::KeyC),
                ('d', Default::default(), KeyCode::KeyD),
                ('e', Default::default(), KeyCode::KeyE),
                ('f', Default::default(), KeyCode::KeyF),
                ('g', Default::default(), KeyCode::KeyG),
                ('h', Default::default(), KeyCode::KeyH),
                ('i', Default::default(), KeyCode::KeyI),
                ('j', Default::default(), KeyCode::KeyJ),
                ('k', Default::default(), KeyCode::KeyK),
                ('l', Default::default(), KeyCode::KeyL),
                ('m', Default::default(), KeyCode::KeyM),
                ('n', Default::default(), KeyCode::KeyN),
                ('o', Default::default(), KeyCode::KeyO),
                ('p', Default::default(), KeyCode::KeyP),
                ('q', Default::default(), KeyCode::KeyQ),
                ('r', Default::default(), KeyCode::KeyR),
                ('s', Default::default(), KeyCode::KeyS),
                ('t', Default::default(), KeyCode::KeyT),
                ('u', Default::default(), KeyCode::KeyU),
                ('v', Default::default(), KeyCode::KeyV),
                ('w', Default::default(), KeyCode::KeyW),
                ('x', Default::default(), KeyCode::KeyX),
                ('y', Default::default(), KeyCode::KeyY),
                ('z', Default::default(), KeyCode::KeyZ),
                ('0', Default::default(), KeyCode::Digit0),
                ('1', Default::default(), KeyCode::Digit1),
                ('2', Default::default(), KeyCode::Digit2),
                ('3', Default::default(), KeyCode::Digit3),
                ('4', Default::default(), KeyCode::Digit4),
                ('5', Default::default(), KeyCode::Digit5),
                ('6', Default::default(), KeyCode::Digit6),
                ('7', Default::default(), KeyCode::Digit7),
                ('8', Default::default(), KeyCode::Digit8),
                ('9', Default::default(), KeyCode::Digit9),
                ('-', Default::default(), KeyCode::Minus),
                ('/', Default::default(), KeyCode::Slash),
            ],
            plus_guard: Default::default(),
            times_guard: Default::default(),
            pow_guard: Default::default(),
            backspace_guard: Default::default(),
            arr_l_guard: Default::default(),
            arr_r_guard: Default::default(),
            dot_guard: Default::default(),
            space_guard: Default::default(),
            force_def_str1: String::from("7.4 a b * * r 2 ^ /"),
            force_def_str2: String::new(),
            force_prog: Vec::new(),
            invalid_prog: false,
            comp_stack: Vec::new(),
        }
    }
    fn compute_force(&mut self, r: f64, a: f64, b: f64) -> f64 {
        self.comp_stack.clear();
        for t in &self.force_prog {
            match t {
                ExprTok::Num(n) => self.comp_stack.push(*n),
                ExprTok::R => self.comp_stack.push(r),
                ExprTok::A => self.comp_stack.push(a),
                ExprTok::B => self.comp_stack.push(b),
                ExprTok::Add => {
                    let y = self.comp_stack.pop().unwrap();
                    let x = self.comp_stack.pop().unwrap();
                    self.comp_stack.push(x + y);
                }
                ExprTok::Sub => {
                    let y = self.comp_stack.pop().unwrap();
                    let x = self.comp_stack.pop().unwrap();
                    self.comp_stack.push(x - y);
                }
                ExprTok::Mul => {
                    let y = self.comp_stack.pop().unwrap();
                    let x = self.comp_stack.pop().unwrap();
                    self.comp_stack.push(x * y);
                }
                ExprTok::Div => {
                    let y = self.comp_stack.pop().unwrap();
                    let x = self.comp_stack.pop().unwrap();
                    self.comp_stack.push(x / y);
                }
                ExprTok::Pow => {
                    let y = self.comp_stack.pop().unwrap();
                    let x = self.comp_stack.pop().unwrap();
                    self.comp_stack.push(x.powf(y));
                }
            }
        }
        self.comp_stack.pop().unwrap_or(0.0)
    }
    fn apply_force(&mut self) {
        for p in self.electrons.iter_mut().chain(self.positrons.iter_mut()) {
            let [x, y] = p.pos;
            let [x, y] = [x as usize, y as usize];
            let mut gradient = [0.0, 0.0];
            for dx in [0, 1] {
                for dy in [0, 1] {
                    if x < dx
                        || x + dx >= SIMULATION_SIZE
                        || y < dy
                        || y + dy >= SIMULATION_SIZE
                    {
                        continue;
                    }
                    if dx == 0 && dy == 0 {
                        continue;
                    } else if dx == 0 {
                        gradient[1] += self.photons[y - dy][x].abs
                            - self.photons[y + dy][x].abs;
                    } else {
                        gradient[0] += self.photons[y][x - dx].abs
                            - self.photons[y][x + dx].abs;
                    }
                }
            }
            p.vel[0] += COUPLING * gradient[0] * self.dt;
            p.vel[1] += COUPLING * gradient[1] * self.dt;
        }
    }
    fn dissipate_photons(&mut self) {
        for x in 0..SIMULATION_SIZE {
            for y in 0..SIMULATION_SIZE {
                self.photons[y][x].abs -=
                    DISSIPATION * self.photons[y][x].abs * self.dt;
            }
        }
    }
    fn diffuse_photons(&mut self) {
        for x in 0..SIMULATION_SIZE as isize {
            for y in 0..SIMULATION_SIZE as isize {
                for dx in [-1, 0, 1] {
                    for dy in [-1, 0, 1] {
                        if x + dx < 0
                            || x + dx >= SIMULATION_SIZE as isize
                            || y + dy < 0
                            || y + dy >= SIMULATION_SIZE as isize
                            || (dx == 0 && dy == 0)
                        {
                            continue;
                        }
                        let that =
                            &self.photons[(y + dy) as usize][(x + dx) as usize];
                        self.photons[y as usize][x as usize].change -= DIFFUSION
                            * (self.photons[y as usize][x as usize].abs
                                - that.abs)
                            * self.dt;
                    }
                }
            }
        }
        for x in 0..SIMULATION_SIZE {
            for y in 0..SIMULATION_SIZE {
                self.photons[y][x].abs += self.photons[y][x].change;
                self.photons[y][x].change = 0.0;
            }
        }
    }
    fn advance(&mut self) {
        for e in &mut self.electrons {
            let [vx, vy] = e.vel;
            let [dx, dy] = [vx * self.dt, vy * self.dt];
            e.pos = [e.pos[0] + dx, e.pos[1] + dy];
        }
        for p in &mut self.positrons {
            let [vx, vy] = p.vel;
            let [dx, dy] = [vx * self.dt, vy * self.dt];
            p.pos = [p.pos[0] + dx, p.pos[1] + dy];
        }
        self.dissipate_photons();
        self.diffuse_photons();
    }
    fn parse_force_prog(&mut self) -> Option<Vec<ExprTok>> {
        let mut new_prog = Vec::new();
        let mut s = self.force_def_str1.clone();
        s += &self.force_def_str2;
        let mut i = 0;
        let mut j = 0;
        let mut op_count = 0;
        while j < s.len() {
            match s.as_bytes()[j] {
                b' ' => {
                    let slc = std::str::from_utf8(&s.as_bytes()[i..j]).unwrap();
                    if !slc.is_empty() {
                        if let Ok(n) = slc.parse::<f64>() {
                            new_prog.push(ExprTok::Num(n));
                        } else if let Ok(n) = slc.parse::<usize>() {
                            new_prog.push(ExprTok::Num(n as f64));
                        } else {
                            println!("no num: `{}`", slc);
                            return None;
                        }
                        op_count += 1;
                    }
                }

                b'r' => {
                    new_prog.push(ExprTok::R);
                    op_count += 1;
                }
                b'a' => {
                    new_prog.push(ExprTok::A);
                    op_count += 1;
                }
                b'b' => {
                    new_prog.push(ExprTok::B);
                    op_count += 1;
                }

                b'+' => {
                    if op_count < 2 {
                        return None;
                    }
                    new_prog.push(ExprTok::Add);
                    op_count -= 1;
                }
                b'-' => {
                    if op_count < 2 {
                        return None;
                    }
                    new_prog.push(ExprTok::Sub);
                    op_count -= 1;
                }
                b'*' => {
                    if op_count < 2 {
                        return None;
                    }
                    new_prog.push(ExprTok::Mul);
                    op_count -= 1;
                }
                b'/' => {
                    if op_count < 2 {
                        return None;
                    }
                    new_prog.push(ExprTok::Div);
                    op_count -= 1;
                }
                b'^' => {
                    if op_count < 2 {
                        return None;
                    }
                    new_prog.push(ExprTok::Pow);
                    op_count -= 1;
                }

                b'.' => {
                    j += 1;
                    continue;
                }

                c => {
                    if c.is_ascii_digit() {
                        j += 1;
                        continue;
                    } else {
                        println!("unknown char");
                        return None;
                    }
                }
            }
            j += 1;
            i = j;
        }
        Some(new_prog)
    }
    fn interact(&mut self) {
        // if let Some(new_prog) = self.parse_force_prog() {
        //     self.force_prog = new_prog;
        //     self.invalid_prog = false;
        // } else {
        //     self.invalid_prog = true;
        // }

        for p in self.electrons.iter().chain(self.positrons.iter()) {
            let [x, y] = p.pos;
            let cell = &mut self.photons[y as usize][x as usize];
            if cell.abs <= 0.0 {
                cell.abs = COUPLING * p.charge;
            } else {
                cell.abs += COUPLING * p.charge;
            }
        }
        self.apply_force();
    }
    fn detect_edge(&mut self) {
        for p in self.electrons.iter_mut().chain(self.positrons.iter_mut()) {
            let [x, y] = &mut p.pos;
            if *x <= 0.0 {
                *x = 0.0;
                p.vel[0] *= -1.0;
            } else if *x >= SIMULATION_SIZE as f64 - 1.0 {
                *x = SIMULATION_SIZE as f64 - 1.0;
                p.vel[0] *= -1.0;
            }
            if *y <= 0.0 {
                *y = 0.0;
                p.vel[1] *= -1.0;
            } else if *y >= SIMULATION_SIZE as f64 - 1.0 {
                *y = SIMULATION_SIZE as f64 - 1.0;
                p.vel[1] *= -1.0;
            }
        }
    }
    fn evolve(&mut self) {
        self.detect_edge();
        self.advance();
        self.detect_edge();
        self.interact();
    }
}

#[derive(Debug, Clone, PartialEq)]
enum InfixProg {
    Text(String),
    Add(Box<Self>, Box<Self>),
    Sub(Box<Self>, Box<Self>),
    Mul(Box<Self>, Box<Self>),
    Div(Box<Self>, Box<Self>),
    Pow(Box<Self>, Box<Self>),
}
fn path(pts: Vec<Point<Px>>) -> Path<Px, false> {
    let mut path = PathBuilder::new(pts[0]);
    for p in &pts[1..] {
        path = path.line_to(*p);
    }
    path.build()
}
impl InfixProg {
    const PAR_WIDTH: f32 = 5.0;
    const DIV_HEIGHT: f32 = 16.0;
    const PLUS_SIZE: f32 = 16.0;
    const MUL_SIZE: f32 = 16.0;
    const OP_PAD: f32 = 14.0;

    fn additive(&self) -> bool {
        match self {
            InfixProg::Add(_, _) => true,
            InfixProg::Sub(_, _) => true,
            _ => false,
        }
    }
    fn par_in_pow(&self) -> bool {
        if let Self::Text(_) = self {
            false
        } else {
            true
        }
    }
    fn width(&self, paren: bool, cx: &mut GraphicsContext) -> f32 {
        let par_w = if paren { Self::PAR_WIDTH * 2.0 } else { 0.0 };
        let inner_w = match self {
            InfixProg::Text(s) => {
                cx.gfx.measure_text::<Px>(s).size.width.into_float()
            }
            InfixProg::Add(a, b) => {
                a.width(false, cx)
                    + b.width(false, cx)
                    + Self::PLUS_SIZE
                    + Self::OP_PAD
            }
            InfixProg::Sub(a, b) => {
                a.width(false, cx)
                    + b.width(b.additive(), cx)
                    + Self::PLUS_SIZE
                    + Self::OP_PAD
            }
            InfixProg::Mul(a, b) => {
                a.width(a.additive(), cx)
                    + b.width(b.additive(), cx)
                    + Self::MUL_SIZE
                    + Self::OP_PAD
            }
            InfixProg::Div(a, b) => a.width(false, cx).max(b.width(false, cx)),
            InfixProg::Pow(a, b) => {
                a.width(a.par_in_pow(), cx) + b.width(false, cx)
            }
        };
        inner_w + par_w
    }
    fn height(&self, cx: &mut GraphicsContext) -> f32 {
        match self {
            InfixProg::Text(s) => {
                cx.gfx.measure_text::<Px>(s).size.height.into_float()
            }
            InfixProg::Add(a, b) => a.height(cx).max(b.height(cx)),
            InfixProg::Sub(a, b) => a.height(cx).max(b.height(cx)),
            InfixProg::Mul(a, b) => a.height(cx).max(b.height(cx)),
            InfixProg::Div(a, b) => {
                a.height(cx) + b.height(cx) + Self::DIV_HEIGHT
            }
            InfixProg::Pow(a, b) => a.height(cx) + b.height(cx),
        }
    }
    fn render_plus(
        top_left: Point<Px>,
        a_w: f32,
        height: f32,
        cx: &mut GraphicsContext,
    ) {
        let center = top_left
            + Size {
                width: Px::from_float(
                    a_w + (Self::OP_PAD + Self::PLUS_SIZE) / 2.0,
                ),
                height: Px::from_float(height / 2.0),
            };
        let mut left = center;
        left.x -= Px::from_float(Self::PLUS_SIZE / 2.0);

        let mut right = center;
        right.x += Px::from_float(Self::PLUS_SIZE / 2.0);

        let mut bot = center;
        bot.y -= Px::from_float(Self::PLUS_SIZE / 2.0);

        let mut top = center;
        top.y += Px::from_float(Self::PLUS_SIZE / 2.0);

        cx.gfx
            .draw_shape(&path(vec![left, right]).stroke(StrokeOptions {
                color: Color::GRAY,
                line_width: Px::from_float(3.0),
                ..Default::default()
            }));
        cx.gfx
            .draw_shape(&path(vec![top, bot]).stroke(StrokeOptions {
                color: Color::GRAY,
                line_width: Px::from_float(3.0),
                ..Default::default()
            }));
    }
    fn render_minus(
        top_left: Point<Px>,
        a_w: f32,
        height: f32,
        cx: &mut GraphicsContext,
    ) {
        let left = top_left
            + Size {
                width: Px::from_float(a_w + Self::OP_PAD / 2.0),
                height: Px::from_float(height / 2.0),
            };
        let right = top_left
            + Size {
                width: Px::from_float(
                    a_w + Self::OP_PAD / 2.0 + Self::PLUS_SIZE,
                ),
                height: Px::from_float(height / 2.0),
            };
        cx.gfx
            .draw_shape(&path(vec![left, right]).stroke(StrokeOptions {
                color: Color::GRAY,
                line_width: Px::from_float(3.0),
                ..Default::default()
            }));
    }
    fn render_mul(
        top_left: Point<Px>,
        a_w: f32,
        height: f32,
        cx: &mut GraphicsContext,
    ) {
        let center = top_left
            + Size {
                width: Px::from_float(
                    a_w + (Self::OP_PAD + Self::MUL_SIZE) / 2.0,
                ),
                height: Px::from_float(height / 2.0),
            };
        let mut tl = center;
        tl.x -= Px::from_float(Self::MUL_SIZE / 2.0);
        tl.y -= Px::from_float(Self::MUL_SIZE / 2.0);

        let mut tr = center;
        tr.x += Px::from_float(Self::MUL_SIZE / 2.0);
        tr.y -= Px::from_float(Self::MUL_SIZE / 2.0);

        let mut bl = center;
        bl.x -= Px::from_float(Self::MUL_SIZE / 2.0);
        bl.y += Px::from_float(Self::MUL_SIZE / 2.0);

        let mut br = center;
        br.x += Px::from_float(Self::MUL_SIZE / 2.0);
        br.y += Px::from_float(Self::MUL_SIZE / 2.0);

        cx.gfx.draw_shape(&path(vec![tl, br]).stroke(StrokeOptions {
            color: Color::GRAY,
            line_width: Px::from_float(3.0),
            ..Default::default()
        }));
        cx.gfx.draw_shape(&path(vec![bl, tr]).stroke(StrokeOptions {
            color: Color::GRAY,
            line_width: Px::from_float(3.0),
            ..Default::default()
        }));
    }
    fn render(
        &self,
        paren: bool,
        mut top_left: Point<Px>,
        cx: &mut GraphicsContext,
    ) {
        let height = self.height(cx);
        let width = self.width(paren, cx);

        if paren {
            cx.gfx.draw_shape(
                &path(vec![
                    top_left
                        + Size {
                            height: Px::from_float(0.0),
                            width: Px::from_float(Self::PAR_WIDTH),
                        },
                    top_left,
                    top_left
                        + Size {
                            height: Px::from_float(height),
                            width: Px::from_float(0.0),
                        },
                    top_left
                        + Size {
                            height: Px::from_float(height),
                            width: Px::from_float(Self::PAR_WIDTH),
                        },
                ])
                .stroke(Color::GRAY),
            );
            cx.gfx.draw_shape(
                &path(vec![
                    top_left
                        + Size {
                            height: Px::from_float(0.0),
                            width: Px::from_float(width - Self::PAR_WIDTH),
                        },
                    top_left
                        + Size {
                            height: Px::from_float(0.0),
                            width: Px::from_float(width),
                        },
                    top_left
                        + Size {
                            height: Px::from_float(height),
                            width: Px::from_float(width),
                        },
                    top_left
                        + Size {
                            height: Px::from_float(height),
                            width: Px::from_float(width - Self::PAR_WIDTH),
                        },
                ])
                .stroke(Color::GRAY),
            );
            top_left += Size {
                width: Px::from_float(Self::PAR_WIDTH),
                height: Px::from_float(0.0),
            };
        }
        match self {
            Self::Text(s) => cx
                .gfx
                .draw_text(Text::new(s, Color::GRAY).translate_by(top_left)),
            Self::Add(a, b) => {
                let a_w = a.width(false, cx);
                let a_h = a.height(cx);
                let b_h = b.height(cx);
                a.render(
                    false,
                    top_left
                        + Size {
                            width: Px::from_float(0.0),
                            height: Px::from_float((height - a_h) / 2.0),
                        },
                    cx,
                );
                Self::render_plus(top_left, a_w, height, cx);
                b.render(
                    false,
                    top_left
                        + Size {
                            width: Px::from_float(
                                a_w + Self::PLUS_SIZE + Self::OP_PAD,
                            ),
                            height: Px::from_float((height - b_h) / 2.0),
                        },
                    cx,
                );
            }
            Self::Sub(a, b) => {
                let a_w = a.width(false, cx);
                let a_h = a.height(cx);
                let b_h = b.height(cx);
                a.render(
                    false,
                    top_left
                        + Size {
                            width: Px::from_float(0.0),
                            height: Px::from_float((height - a_h) / 2.0),
                        },
                    cx,
                );
                Self::render_minus(top_left, a_w, height, cx);
                b.render(
                    b.additive(),
                    top_left
                        + Size {
                            width: Px::from_float(
                                a_w + Self::PLUS_SIZE + Self::OP_PAD,
                            ),
                            height: Px::from_float((height - b_h) / 2.0),
                        },
                    cx,
                );
            }
            Self::Mul(a, b) => {
                let a_w = a.width(a.additive(), cx);
                let a_h = a.height(cx);
                let b_h = b.height(cx);
                a.render(
                    a.additive(),
                    top_left
                        + Size {
                            width: Px::from_float(0.0),
                            height: Px::from_float((height - a_h) / 2.0),
                        },
                    cx,
                );
                Self::render_mul(top_left, a_w, height, cx);
                b.render(
                    b.additive(),
                    top_left
                        + Size {
                            width: Px::from_float(
                                a_w + Self::MUL_SIZE + Self::OP_PAD,
                            ),
                            height: Px::from_float((height - b_h) / 2.0),
                        },
                    cx,
                );
            }
            Self::Div(a, b) => {
                let a_w = a.width(false, cx);
                let b_w = b.width(false, cx);
                let a_h = a.height(cx);
                a.render(
                    false,
                    top_left
                        + Size {
                            width: Px::from_float((width - a_w) / 2.0),
                            height: Px::from_float(0.0),
                        },
                    cx,
                );
                cx.gfx.draw_shape(
                    &path(vec![
                        top_left
                            + Size {
                                width: Px::from_float(0.0),
                                height: Px::from_float(
                                    a_h + Self::DIV_HEIGHT / 2.0,
                                ),
                            },
                        top_left
                            + Size {
                                width: Px::from_float(a_w.max(b_w)),
                                height: Px::from_float(
                                    a_h + Self::DIV_HEIGHT / 2.0,
                                ),
                            },
                    ])
                    .stroke(Color::GRAY),
                );
                b.render(
                    false,
                    top_left
                        + Size {
                            width: Px::from_float((width - b_w) / 2.0),
                            height: Px::from_float(a_h + Self::DIV_HEIGHT),
                        },
                    cx,
                );
            }
            Self::Pow(a, b) => {
                let a_w = a.width(a.par_in_pow(), cx);
                let b_h = b.height(cx);
                a.render(
                    false,
                    top_left
                        + Size {
                            width: Px::from_float(0.0),
                            height: Px::from_float(b_h),
                        },
                    cx,
                );
                b.render(
                    false,
                    top_left
                        + Size {
                            width: Px::from_float(a_w),
                            height: Px::from_float(0.0),
                        },
                    cx,
                );
            }
        }
    }
}

// Drawing routines
//
impl ParticleSystem {
    fn draw(&mut self, cx: &mut GraphicsContext) {
        self.draw_fields(cx);
        self.draw_particles(cx);
        self.draw_force_str(cx);
        self.draw_force_pretty(cx);
    }
    fn draw_particles(&mut self, cx: &mut GraphicsContext) {
        let sim_size = SIMULATION_SIZE as f64;

        for p in self.electrons.iter().chain(self.positrons.iter()) {
            let [x, y] = p.pos;

            let width = cx.gfx.size().width.into_float();
            let height = cx.gfx.size().height.into_float();
            let pos = Point::px(
                width * (x / sim_size) as f32,
                height * (y / sim_size) as f32,
            );
            let color = if p.charge > 0.0 {
                Color::RED
            } else {
                Color::BLUE
            };
            cx.gfx.draw_shape(
                Shape::filled_circle(
                    Px::from_float(
                        (width + height) / 4.0 * (1.0 / sim_size) as f32,
                    ),
                    color,
                    cushy::kludgine::Origin::Center,
                )
                .translate_by(pos),
            );
        }
    }
    fn draw_fields(&mut self, cx: &mut GraphicsContext) {
        let width = cx.gfx.size().width.into_float();
        let height = cx.gfx.size().height.into_float();

        for x in 0..SIMULATION_SIZE {
            for y in 0..SIMULATION_SIZE {
                let intensity = (self.photons[y][x].abs / COUPLING) as f32;
                let color = if intensity >= 0.0 {
                    Color::new_f32(1.0 - intensity, 1.0, 1.0, 1.0)
                } else {
                    Color::new_f32(1.0, 1.0, 1.0 + intensity, 1.0)
                };
                cx.gfx.draw_shape(
                    Shape::filled_rect(
                        Rect::new(
                            Point::px(
                                (x as f32) * width / SIMULATION_SIZE as f32,
                                (y as f32) * height / SIMULATION_SIZE as f32,
                            ),
                            Size::px(
                                width / SIMULATION_SIZE as f32,
                                height / SIMULATION_SIZE as f32,
                            ),
                        ),
                        color,
                    )
                    .translate_by(Point::px(0.0, 0.0)),
                );
            }
        }
    }
    fn draw_force_str(&mut self, cx: &mut GraphicsContext) {
        let color = if self.invalid_prog {
            Color::CORAL
        } else {
            Color::GRAY
        };
        cx.gfx.draw_text(
            Text::new(&self.force_def_str1, color)
                .translate_by(Point::px(2.0 * MARGIN, 0)),
        );
        let str1_dims = cx.gfx.measure_text::<Px>(&self.force_def_str1);
        let width = str1_dims.size.width;
        let origin = (str1_dims.ascent + str1_dims.descent) / 2;
        let height = str1_dims.ascent + str1_dims.descent;

        cx.gfx.draw_shape(
            Shape::filled_rect(
                Rect::new(
                    Point::px(width, origin),
                    Size {
                        width: Px::new(3),
                        height,
                    },
                ),
                color,
            )
            .translate_by(Point::px(2.0 * MARGIN, 0)),
        );
        cx.gfx.draw_text(
            Text::new(&self.force_def_str2, color).translate_by(Point::px(
                width.into_float() + 2.0 * MARGIN,
                0.0,
            )),
        );

        self.update_keys(cx);
    }
    fn draw_force_pretty(&mut self, cx: &mut GraphicsContext) {
        if self.force_prog.is_empty() {
            return;
        }
        let mut stack = vec![];
        for tok in &self.force_prog {
            match tok {
                ExprTok::Num(n) => stack.push(InfixProg::Text(n.to_string())),
                ExprTok::R => stack.push(InfixProg::Text("r".to_string())),
                ExprTok::A => stack.push(InfixProg::Text("a".to_string())),
                ExprTok::B => stack.push(InfixProg::Text("b".to_string())),
                ExprTok::Add => {
                    let b = stack.pop().unwrap();
                    let a = stack.pop().unwrap();
                    stack.push(InfixProg::Add(Box::new(a), Box::new(b)));
                }
                ExprTok::Sub => {
                    let b = stack.pop().unwrap();
                    let a = stack.pop().unwrap();
                    stack.push(InfixProg::Sub(Box::new(a), Box::new(b)));
                }
                ExprTok::Mul => {
                    let b = stack.pop().unwrap();
                    let a = stack.pop().unwrap();
                    stack.push(InfixProg::Mul(Box::new(a), Box::new(b)));
                }
                ExprTok::Div => {
                    let b = stack.pop().unwrap();
                    let a = stack.pop().unwrap();
                    stack.push(InfixProg::Div(Box::new(a), Box::new(b)));
                }
                ExprTok::Pow => {
                    let b = stack.pop().unwrap();
                    let a = stack.pop().unwrap();
                    stack.push(InfixProg::Pow(Box::new(a), Box::new(b)));
                }
            }
        }

        stack.last().unwrap().render(
            false,
            Point::px(
                30.0 + 2.0 * MARGIN,
                cx.gfx.size().height.into_float() / 2.0,
            ),
            cx,
        );
    }
    fn update_keys(&mut self, cx: &mut GraphicsContext) {
        if cx.modifiers().lshift_state() == ModifiersKeyState::Pressed
            || cx.modifiers().rshift_state() == ModifiersKeyState::Pressed
        {
            if cx.key_pressed(KeyCode::Equal) {
                if !self.plus_guard.guard {
                    self.force_def_str1.push('+');
                    self.plus_guard.guard = true;
                }
            } else {
                self.plus_guard.guard = false;
            }
            if cx.key_pressed(KeyCode::Digit8) {
                if !self.times_guard.guard {
                    self.force_def_str1.push('*');
                    self.times_guard.guard = true;
                }
            } else {
                self.times_guard.guard = false;
            }
            if cx.key_pressed(KeyCode::Digit6) {
                if !self.pow_guard.guard {
                    self.force_def_str1.push('^');
                    self.pow_guard.guard = true;
                }
            } else {
                self.pow_guard.guard = false;
            }
        } else {
            for (c, guard, k) in &mut self.keyboard {
                if cx.key_pressed(*k) {
                    if !guard.guard {
                        self.force_def_str1.push(*c);
                        guard.guard = true;
                    }
                } else {
                    guard.guard = false;
                }
            }
            if cx.key_pressed(KeyCode::Backspace) {
                if !self.backspace_guard.guard {
                    self.force_def_str1.pop();
                    self.backspace_guard.guard = true;
                }
            } else {
                self.backspace_guard.guard = false;
            }
            if cx.key_pressed(KeyCode::ArrowLeft) {
                if !self.arr_l_guard.guard {
                    if let Some(c) = self.force_def_str1.pop() {
                        self.force_def_str2.insert(0, c);
                    }
                    self.arr_l_guard.guard = true;
                }
            } else {
                self.arr_l_guard.guard = false;
            }
            if cx.key_pressed(KeyCode::ArrowRight) {
                if !self.arr_r_guard.guard {
                    if self.force_def_str2.len() > 0 {
                        let c = self.force_def_str2.remove(0);
                        self.force_def_str1.push(c);
                    }
                    self.arr_r_guard.guard = true;
                }
            } else {
                self.arr_r_guard.guard = false;
            }
            if cx.key_pressed(KeyCode::Period) {
                if !self.dot_guard.guard {
                    self.force_def_str1.push('.');
                    self.dot_guard.guard = true;
                }
            } else {
                self.dot_guard.guard = false;
            }
            if cx.key_pressed(KeyCode::Space) {
                if !self.space_guard.guard {
                    self.force_def_str1.push(' ');
                    self.space_guard.guard = true;
                }
            } else {
                self.space_guard.guard = false;
            }
        }
    }
}

fn main() -> cushy::Result<()> {
    let mut sys = ParticleSystem::new();

    Canvas::new(move |cx| {
        sys.evolve();
        sys.draw(cx);
    })
    .tick(Tick::redraws_per_second(60))
    .run()
}
