use cushy::{
    context::GraphicsContext,
    kludgine::{
        app::winit::{
            event::MouseButton,
            keyboard::{KeyCode, ModifiersKeyState},
        },
        figures::{units::Px, FloatConversion, Point, Px2D, Rect, Size},
        shapes::{Path, PathBuilder, Shape, StrokeOptions},
        text::Text,
        Color, DrawableExt,
    },
    widgets::Canvas,
    Run, Tick,
};
use rand::Rng;

const PARTICLE_COUNT: usize = 2;
const SIMULATION_SIZE: f64 = 100.0;
const DENSITY_PLY: usize = 7;
const MARGIN: f32 = 50.0;
const INIT_VEL: f64 = 0.1e1;

#[derive(Debug, Default, Clone, Copy, PartialEq)]
struct Particle {
    pos: [f64; 2],
    vel: [f64; 2],
    charge: f64,
}

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

struct ParticleSystem {
    particles: Vec<Particle>,
    particle_r: f64,
    speed_lim: f64,
    force_r: f64,
    mag: f64,
    dt: f64,
    density: GuardedBool,
    temper: GuardedBool,
    add_part: GuardedBool,
    del_part: GuardedBool,
    keyboard: [(char, GuardedBool, KeyCode); 38],
    plus_guard: GuardedBool,
    times_guard: GuardedBool,
    pow_guard: GuardedBool,
    backspace_guard: GuardedBool,
    arr_l_guard: GuardedBool,
    arr_r_guard: GuardedBool,
    dot_guard: GuardedBool,
    space_guard: GuardedBool,
    ambient_mag: f64,
    randomize_guard: bool,
    force_def_str1: String, // Before cursor
    force_def_str2: String, // After cursor
    force_prog: Vec<ExprTok>,
    invalid_prog: bool,
    comp_stack: Vec<f64>,
}

// Particle behaviour
//
impl ParticleSystem {
    fn random_particles() -> [Particle; PARTICLE_COUNT] {
        let mut particles = [Particle::default(); PARTICLE_COUNT];
        let mut rng = rand::thread_rng();
        for i in 0..PARTICLE_COUNT {
            let x = rng.gen_range(0.0..SIMULATION_SIZE);
            let y = rng.gen_range(0.0..SIMULATION_SIZE);
            let vx = rng.gen_range(-INIT_VEL..=INIT_VEL);
            let vy = rng.gen_range(-INIT_VEL..=INIT_VEL);
            let charge = if i % 2 == 0 { 1.0 } else { -1.0 };
            let p = Particle {
                pos: [x, y],
                vel: [vx, vy],
                charge,
            };
            particles[i] = p;
        }
        particles
    }
    fn new() -> Self {
        // let particles = Self::random_particles().into();
        let particles = vec![
            Particle {
                pos: [SIMULATION_SIZE / 2.0, SIMULATION_SIZE / 4.0],
                vel: [INIT_VEL / 2.0, 0.0],
                charge: 1.0,
            },
            Particle {
                pos: [SIMULATION_SIZE / 2.0, 3.0 * SIMULATION_SIZE / 4.0],
                vel: [-INIT_VEL / 2.0, 0.0],
                charge: -1.0,
            },
        ];
        Self {
            particles,
            particle_r: 1.0,
            speed_lim: 20.0,
            force_r: SIMULATION_SIZE,
            mag: 0.0,
            dt: 0.3,
            density: Default::default(),
            temper: Default::default(),
            add_part: Default::default(),
            del_part: Default::default(),
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
            ambient_mag: 0.0,
            randomize_guard: false,
            force_def_str1: String::from("7.4 a b * * r 2 ^ /"),
            force_def_str2: String::new(),
            force_prog: Vec::new(),
            invalid_prog: false,
            comp_stack: Vec::new(),
        }
    }
    fn bounce_x(&mut self, p_idx: usize) {
        self.particles[p_idx].vel[0] *= -1.0;
    }
    fn bounce_y(&mut self, p_idx: usize) {
        self.particles[p_idx].vel[1] *= -1.0;
    }
    fn compute_force(&mut self, r: f64, a: f64, b: f64) -> f64 {
        self.comp_stack.clear();
        println!("{:?}", &self.force_prog);
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
    fn apply_force(&mut self, i: usize, j: usize) {
        let [xi, yi] = self.particles[i].pos;
        let [xj, yj] = self.particles[j].pos;
        let [dx, dy] = [xi - xj, yi - yj];
        let r = (dx * dx + dy * dy).sqrt();

        let a = self.particles[i].charge;
        let b = self.particles[j].charge;

        let f_mag = self.compute_force(r, a, b);
        let [vxi, vyi] = &mut self.particles[i].vel;

        *vxi += f_mag * dx.signum();
        *vyi += f_mag * dy.signum();

        let [vxj, vyj] = &mut self.particles[j].vel;
        *vxj -= f_mag * dx.signum();
        *vyj -= f_mag * dy.signum();
    }
    fn apply_magnetism(&mut self, i: usize, j: usize) {
        let [xi, yi] = self.particles[i].pos;
        let [xj, yj] = self.particles[j].pos;
        let [dx, dy] = [xi - xj, yi - yj];
        let r = (dx * dx + dy * dy).sqrt();

        let ci = self.particles[i].charge;
        let cj = self.particles[j].charge;

        let scale = self.mag * ci * cj / (r * r);

        let [vxi, vyi] = self.particles[i].vel;
        let [vxj, vyj] = self.particles[j].vel;

        let vi_r_dot = vxi * dx + vyi * dy;
        let vj_r_dot = vxj * dx + vyj * dy;
        let vi_vj_dot = vxi * vxj + vyi * vyj;

        let dir_i_x = vxj * vi_r_dot - dx * vi_vj_dot;
        let dir_i_y = vyj * vi_r_dot - dy * vi_vj_dot;
        let dir_j_x = vxi * vj_r_dot - dx * vi_vj_dot;
        let dir_j_y = vyi * vj_r_dot - dy * vi_vj_dot;

        let [vxi, vyi] = &mut self.particles[i].vel;
        *vxi += scale * dir_i_x * self.dt;
        *vyi += scale * dir_i_y * self.dt;

        let [vxj, vyj] = &mut self.particles[j].vel;
        *vxj += scale * dir_j_x * self.dt;
        *vyj += scale * dir_j_y * self.dt;
    }
    fn apply_ambient_mag(&mut self, i: usize) {
        let [vx, vy] = self.particles[i].vel;
        let charge = self.particles[i].charge;

        let dvx = vy * charge * self.ambient_mag;
        let dvy = -vx * charge * self.ambient_mag;

        let [vx, vy] = &mut self.particles[i].vel;
        *vx += dvx;
        *vy += dvy;
    }
    fn advance_particles(&mut self) {
        for p in &mut self.particles {
            let [vx, vy] = p.vel;

            let [dx, dy] = [vx * self.dt, vy * self.dt];
            let [x, y] = &mut p.pos;
            *x += dx;
            *y += dy;
        }
    }
    fn particle_speed(&self, i: usize) -> f64 {
        let p = self.particles[i];
        (p.vel[0] * p.vel[0] + p.vel[1] * p.vel[1]).sqrt()
    }
    fn detect_edge_collisions(&mut self) {
        for i in 0..self.particles.len() {
            let mut bounce_x = false;
            let mut bounce_y = false;
            let [x, y] = &mut self.particles[i].pos;
            if *x <= self.particle_r {
                *x = self.particle_r;
                bounce_x = true;
            }
            if *x >= SIMULATION_SIZE - self.particle_r {
                *x = SIMULATION_SIZE - self.particle_r;
                bounce_x = true;
            }
            if *y <= self.particle_r {
                *y = self.particle_r;
                bounce_y = true;
            }
            if *y >= SIMULATION_SIZE - self.particle_r {
                *y = SIMULATION_SIZE - self.particle_r;
                bounce_y = true;
            }
            if bounce_x {
                self.bounce_x(i);
            }
            if bounce_y {
                self.bounce_y(i);
            }
        }
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
    fn particles_interact(&mut self) {
        if let Some(new_prog) = self.parse_force_prog() {
            self.force_prog = new_prog;
            self.invalid_prog = false;
        } else {
            self.invalid_prog = true;
        }
        for i in 0..self.particles.len() {
            self.apply_ambient_mag(i);
            for j in (i + 1)..self.particles.len() {
                let [xi, yi] = self.particles[i].pos;
                let [xj, yj] = self.particles[j].pos;
                let [dx, dy] = [xi - xj, yi - yj];
                let r = (dx * dx + dy * dy).sqrt();
                if r < self.force_r && r > 2.0 * self.particle_r {
                    self.apply_force(i, j);
                    if self.particle_speed(i) < self.speed_lim
                        && self.particle_speed(j) < self.speed_lim
                    {
                        self.apply_magnetism(i, j);
                    }
                }
            }
        }
    }
    fn barycenter(&mut self) {
        let mut x = 0.0;
        let mut y = 0.0;
        let mut vx = 0.0;
        let mut vy = 0.0;
        for p in &self.particles {
            x += p.pos[0];
            y += p.pos[1];
            vx += p.vel[0];
            vy += p.vel[1];
        }
        x /= self.particles.len() as f64;
        y /= self.particles.len() as f64;
        vx /= self.particles.len() as f64;
        vy /= self.particles.len() as f64;
        for p in &mut self.particles {
            p.pos[0] = SIMULATION_SIZE / 2.0 - (x - p.pos[0]);
            p.pos[1] = SIMULATION_SIZE / 2.0 - (y - p.pos[1]);
            p.vel[0] -= vx;
            p.vel[1] -= vy;
        }
    }
    fn evolve(&mut self) {
        self.barycenter();
        self.advance_particles();
        self.particles_interact();
        // self.detect_edge_collisions();
    }

    fn add_particle(&mut self) {
        let charge = if let Some(p) = self.particles.last() {
            -p.charge
        } else {
            1.0
        };
        let mut rng = rand::thread_rng();
        let x = rng.gen_range(0.0..SIMULATION_SIZE);
        let y = rng.gen_range(0.0..SIMULATION_SIZE);
        let vx = rng.gen_range(-INIT_VEL..=INIT_VEL);
        let vy = rng.gen_range(-INIT_VEL..=INIT_VEL);

        self.particles.push(Particle {
            pos: [x, y],
            vel: [vx, vy],
            charge,
        });
    }
    fn del_particle(&mut self) {
        self.particles.pop();
    }
    fn randomize(&mut self) {
        let mut rng = rand::thread_rng();
        for p in &mut self.particles {
            let x = rng.gen_range(0.0..SIMULATION_SIZE);
            let y = rng.gen_range(0.0..SIMULATION_SIZE);
            let vx = rng.gen_range(-INIT_VEL..=INIT_VEL);
            let vy = rng.gen_range(-INIT_VEL..=INIT_VEL);
            p.pos = [x, y];
            p.vel = [vx, vy];
        }
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
        let mut density = [[0; DENSITY_PLY + 1]; DENSITY_PLY + 1];
        let mut temper = vec![vec![vec![]; DENSITY_PLY + 1]; DENSITY_PLY + 1];
        for (idx, p) in self.particles.iter().enumerate() {
            let [x, y] = p.pos;
            let i = ((y / SIMULATION_SIZE) * (DENSITY_PLY as f64)) as usize;
            let j = ((x / SIMULATION_SIZE) * (DENSITY_PLY as f64)) as usize;
            if self.density.on {
                density[i][j] += 1;
            }
            if self.temper.on {
                temper[i][j].push(self.particle_speed(idx));
            }

            let width = cx.gfx.size().width.into_float();
            let height = cx.gfx.size().height.into_float();
            let pos = Point::px(
                width * (x / SIMULATION_SIZE) as f32,
                (height - MARGIN) * (y / SIMULATION_SIZE) as f32,
            );
            let color = if p.charge > 0.0 {
                Color::RED
            } else {
                Color::BLUE
            };
            cx.gfx.draw_shape(
                Shape::filled_circle(
                    Px::from_float(
                        (width + height) / 4.0
                            * (self.particle_r / SIMULATION_SIZE) as f32,
                    ),
                    color,
                    cushy::kludgine::Origin::Center,
                )
                .translate_by(pos),
            );
        }
        self.draw_controls(cx);
        if self.density.on {
            self.draw_density(cx, density);
        }
        if self.temper.on {
            self.draw_temper(cx, temper);
        }
        self.draw_force_str(cx);
        self.draw_force_pretty(cx);
    }
    fn draw_controls(&mut self, cx: &mut GraphicsContext) {
        let height = cx.gfx.size().height.into_float();
        let width = cx.gfx.size().width.into_float();
        cx.gfx.draw_shape(
            Shape::filled_rect(
                Rect::new(
                    Point::px(0.0, height - MARGIN),
                    Size::px(width, MARGIN),
                ),
                Color::new_f32(0.8, 0.8, 0.8, 1.0),
            )
            .translate_by(Point::px(0, 0)),
        );
        let mut pos = 10.0;
        self.draw_density_btn(cx, &mut pos);
        pos += 10.0;
        self.draw_temper_btn(cx, &mut pos);
        pos += 10.0;
        self.draw_add_part_btn(cx, &mut pos);
        pos += 10.0;
        self.draw_del_part_btn(cx, &mut pos);
        pos += 10.0;
        self.draw_randomize_btn(cx, &mut pos);
    }
    fn draw_force_str(&mut self, cx: &mut GraphicsContext) {
        let color = if self.invalid_prog {
            Color::CORAL
        } else {
            Color::GRAY
        };
        cx.gfx.draw_text(
            Text::new(&self.force_def_str1, color)
                .translate_by(Point::px(0, 0)),
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
            .translate_by(Point::px(0, 0)),
        );
        cx.gfx.draw_text(
            Text::new(&self.force_def_str2, color)
                .translate_by(Point::px(width, 0)),
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
            Point::px(30.0, cx.gfx.size().height.into_float() / 2.0),
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
    fn draw_density_btn(&mut self, cx: &mut GraphicsContext, x_pos: &mut f32) {
        let size = 120.0;
        let height = cx.gfx.size().height.into_float();
        let density_box = Rect::new(
            Point::px(*x_pos - 5.0, height - 9.0 / 10.0 * MARGIN),
            Size::px(size, MARGIN),
        );
        cx.gfx.draw_shape(&Shape::filled_rect(
            density_box,
            Color::new_f32(0.6, 0.6, 0.6, 1.0),
        ));
        cx.gfx.draw_text(
            Text::new(
                "Density",
                if self.density.on {
                    Color::GREEN
                } else {
                    Color::RED
                },
            )
            .translate_by(Point::px(*x_pos, height - 9.0 / 10.0 * MARGIN)),
        );
        *x_pos += size;
        if let Some(pos) = cx.cursor_position() {
            if density_box.contains(pos)
                && cx.mouse_button_pressed(MouseButton::Left)
            {
                if !self.density.guard {
                    self.density.on = !self.density.on;
                    self.density.guard = true;
                }
            } else {
                self.density.guard = false;
            }
        }
    }
    fn draw_temper_btn(&mut self, cx: &mut GraphicsContext, x_pos: &mut f32) {
        let size = 200.0;
        let height = cx.gfx.size().height.into_float();
        let temper_box = Rect::new(
            Point::px(*x_pos - 5.0, height - 9.0 / 10.0 * MARGIN),
            Size::px(size, MARGIN),
        );
        cx.gfx.draw_shape(&Shape::filled_rect(
            temper_box,
            Color::new_f32(0.6, 0.6, 0.6, 1.0),
        ));
        cx.gfx.draw_text(
            Text::new(
                "Temperature",
                if self.temper.on {
                    Color::GREEN
                } else {
                    Color::RED
                },
            )
            .translate_by(Point::px(*x_pos, height - 9.0 / 10.0 * MARGIN)),
        );
        *x_pos += size;
        if let Some(pos) = cx.cursor_position() {
            if temper_box.contains(pos)
                && cx.mouse_button_pressed(MouseButton::Left)
            {
                if !self.temper.guard {
                    self.temper.on = !self.temper.on;
                    self.temper.guard = true;
                }
            } else {
                self.temper.guard = false;
            }
        }
    }
    fn draw_add_part_btn(&mut self, cx: &mut GraphicsContext, x_pos: &mut f32) {
        let size = 200.0;
        let height = cx.gfx.size().height.into_float();
        let add_part_box = Rect::new(
            Point::px(*x_pos - 5.0, height - 9.0 / 10.0 * MARGIN),
            Size::px(size, MARGIN),
        );
        cx.gfx.draw_shape(&Shape::filled_rect(
            add_part_box,
            Color::new_f32(0.6, 0.6, 0.6, 1.0),
        ));
        cx.gfx.draw_text(
            Text::new("Add Particle", Color::BLUE)
                .translate_by(Point::px(*x_pos, height - 9.0 / 10.0 * MARGIN)),
        );
        *x_pos += size;
        if let Some(pos) = cx.cursor_position() {
            if add_part_box.contains(pos)
                && cx.mouse_button_pressed(MouseButton::Left)
            {
                if !self.add_part.guard {
                    self.add_particle();
                    self.add_part.guard = true;
                }
            } else {
                self.add_part.guard = false;
            }
        }
    }
    fn draw_del_part_btn(&mut self, cx: &mut GraphicsContext, x_pos: &mut f32) {
        let size = 200.0;
        let height = cx.gfx.size().height.into_float();
        let del_part_box = Rect::new(
            Point::px(*x_pos - 5.0, height - 9.0 / 10.0 * MARGIN),
            Size::px(size, MARGIN),
        );
        cx.gfx.draw_shape(&Shape::filled_rect(
            del_part_box,
            Color::new_f32(0.6, 0.6, 0.6, 1.0),
        ));
        cx.gfx.draw_text(
            Text::new("Del Particle", Color::BLUE)
                .translate_by(Point::px(*x_pos, height - 9.0 / 10.0 * MARGIN)),
        );
        *x_pos += size;
        if let Some(pos) = cx.cursor_position() {
            if del_part_box.contains(pos)
                && cx.mouse_button_pressed(MouseButton::Left)
            {
                if !self.del_part.guard {
                    self.del_particle();
                    self.del_part.guard = true;
                }
            } else {
                self.del_part.guard = false;
            }
        }
    }
    fn draw_randomize_btn(
        &mut self,
        cx: &mut GraphicsContext,
        x_pos: &mut f32,
    ) {
        let size = 170.0;
        let height = cx.gfx.size().height.into_float();
        let randomize_box = Rect::new(
            Point::px(*x_pos - 5.0, height - 9.0 / 10.0 * MARGIN),
            Size::px(size, MARGIN),
        );
        cx.gfx.draw_shape(&Shape::filled_rect(
            randomize_box,
            Color::new_f32(0.6, 0.6, 0.6, 1.0),
        ));
        cx.gfx.draw_text(
            Text::new("Randomize", Color::GRAY)
                .translate_by(Point::px(*x_pos, height - 9.0 / 10.0 * MARGIN)),
        );
        *x_pos += size;
        if let Some(pos) = cx.cursor_position() {
            if randomize_box.contains(pos)
                && cx.mouse_button_pressed(MouseButton::Left)
            {
                if !self.randomize_guard {
                    self.randomize();
                    self.randomize_guard = true;
                }
            } else {
                self.randomize_guard = false;
            }
        }
    }
    fn draw_density(
        &mut self,
        cx: &mut GraphicsContext,
        density: [[usize; DENSITY_PLY + 1]; DENSITY_PLY + 1],
    ) {
        let width = cx.gfx.size().width.into_float();
        let height = cx.gfx.size().height.into_float() - MARGIN;
        let dens_max = *density.iter().flatten().max().unwrap();
        for x in 0..DENSITY_PLY + 1 {
            for y in 0..DENSITY_PLY + 1 {
                let x_unit = width / DENSITY_PLY as f32;
                let y_unit = height / DENSITY_PLY as f32;

                let top_left = Point::px(x_unit * x as f32, y_unit * y as f32);

                let shade = density[y][x] as f32 / dens_max as f32;
                cx.gfx.draw_shape(
                    Shape::filled_rect(
                        Rect::new(top_left, Size::px(x_unit, y_unit)),
                        Color::new_f32(0.0, 0.0, 1.0, 0.7 * shade),
                    )
                    .translate_by(Point::px(0, 0)),
                );
            }
        }
    }
    fn draw_temper(
        &mut self,
        cx: &mut GraphicsContext,
        temper: Vec<Vec<Vec<f64>>>,
    ) {
        let width = cx.gfx.size().width.into_float();
        let height = cx.gfx.size().height.into_float() - MARGIN;
        let temper: Vec<Vec<f64>> = temper
            .iter()
            .map(|col| {
                col.iter()
                    .map(|vels| {
                        vels.iter().copied().sum::<f64>() / vels.len() as f64
                    })
                    .collect()
            })
            .collect();
        for x in 0..DENSITY_PLY + 1 {
            for y in 0..DENSITY_PLY {
                let x_unit = width / DENSITY_PLY as f32;
                let y_unit = height / DENSITY_PLY as f32;

                let top_left = Point::px(x_unit * x as f32, y_unit * y as f32);

                let shade =
                    temper[y][x] as f32 / (0.75 * self.speed_lim) as f32;
                cx.gfx.draw_shape(
                    Shape::filled_rect(
                        Rect::new(top_left, Size::px(x_unit, y_unit)),
                        Color::new_f32(1.0, 0.0, 0.0, 0.7 * shade),
                    )
                    .translate_by(Point::px(0, 0)),
                );
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
