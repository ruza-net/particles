use cushy::{
    context::GraphicsContext,
    kludgine::{
        app::winit::event::MouseButton,
        figures::{units::Px, FloatConversion, Point, Px2D, Rect, Size},
        shapes::Shape,
        text::Text,
        Color, DrawableExt,
    },
    widgets::Canvas,
    Run, Tick,
};
use rand::Rng;

const PARTICLE_COUNT: usize = 500;
const SIMULATION_SIZE: f64 = 100.0;
const DENSITY_PLY: usize = 7;
const MARGIN: f32 = 50.0;

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

struct ParticleSystem {
    particles: [Particle; PARTICLE_COUNT],
    particle_r: f64,
    speed_lim: f64,
    force_r: f64,
    force: f64,
    mag: f64,
    dt: f64,
    density: GuardedBool,
    temper: GuardedBool,
    ambient_mag: f64,
    randomize_guard: bool,
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
            let vx = rng.gen_range(-1.0..=1.0);
            let vy = rng.gen_range(-1.0..=1.0);
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
        let particles = Self::random_particles();
        Self {
            particles,
            particle_r: 1.0,
            speed_lim: 20.0,
            force_r: SIMULATION_SIZE,
            force: 2.5,
            mag: 0.1,
            dt: 0.1,
            density: Default::default(),
            temper: Default::default(),
            ambient_mag: 0.001,
            randomize_guard: false,
        }
    }
    fn bounce_x(&mut self, p_idx: usize) {
        self.particles[p_idx].vel[0] *= -1.0;
    }
    fn bounce_y(&mut self, p_idx: usize) {
        self.particles[p_idx].vel[1] *= -1.0;
    }
    fn apply_force(&mut self, i: usize, j: usize) {
        let [xi, yi] = self.particles[i].pos;
        let [xj, yj] = self.particles[j].pos;
        let [dx, dy] = [xi - xj, yi - yj];
        let r = (dx * dx + dy * dy).sqrt();

        let ci = self.particles[i].charge;
        let cj = self.particles[j].charge;

        let [vxi, vyi] = &mut self.particles[i].vel;

        *vxi += ci * cj * self.dt * self.force / (r * r) * dx.signum();
        *vyi += ci * cj * self.dt * self.force / (r * r) * dy.signum();

        let [vxj, vyj] = &mut self.particles[j].vel;
        *vxj -= ci * cj * self.dt * self.force / (r * r) * dx.signum();
        *vyj -= ci * cj * self.dt * self.force / (r * r) * dy.signum();
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
    fn particles_interact(&mut self) {
        for i in 0..PARTICLE_COUNT {
            self.apply_ambient_mag(i);
            for j in (i + 1)..PARTICLE_COUNT {
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
    fn evolve(&mut self) {
        self.advance_particles();
        self.particles_interact();
        self.detect_edge_collisions();
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
        self.draw_randomize_btn(cx, &mut pos);
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
                    self.particles = Self::random_particles();
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
