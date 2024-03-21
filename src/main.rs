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

const PARTICLE_COUNT: usize = 1000;
const SIMULATION_SIZE: f64 = 100.0;
const DENSITY_PLY: usize = 7;
const MARGIN: f32 = 50.0;

#[derive(Debug, Default, Clone, Copy, PartialEq)]
struct Particle {
    pos: [f64; 2],
    vel: [f64; 2],
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct GuardedBool {
    on: bool,
    guard: bool,
}

struct ParticleSystem {
    particles: [Particle; PARTICLE_COUNT],
    particle_r: f64,
    force_r: f64,
    force: f64,
    simulation_dt: f64,
    collisions: GuardedBool,
    heating: GuardedBool,
    density: GuardedBool,
    temper: GuardedBool,
    vd_waals: GuardedBool,
    randomize_guard: bool,
    heating_rate: f64,
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
            let p = Particle {
                pos: [x, y],
                vel: [vx, vy],
            };
            particles[i] = p;
        }
        particles
    }
    fn new() -> Self {
        Self {
            particles: Self::random_particles(),
            particle_r: 1.0,
            force_r: 10.0,
            force: 0.007,
            simulation_dt: 0.3,
            collisions: Default::default(),
            heating: Default::default(),
            density: Default::default(),
            temper: Default::default(),
            vd_waals: Default::default(),
            randomize_guard: false,
            heating_rate: 0.07,
        }
    }
    fn bounce_x(&mut self, p_idx: usize) {
        self.particles[p_idx].vel[0] *= -1.0;
    }
    fn bounce_y(&mut self, p_idx: usize) {
        self.particles[p_idx].vel[1] *= -1.0;
    }
    fn bounce(&mut self, i: usize, j: usize) {
        let [vx_i, vy_i] = self.particles[i].vel;
        let [vx_j, vy_j] = self.particles[j].vel;

        if vx_i.signum() == -vx_j.signum() {
            self.particles[i].vel[0] = vx_j;
            self.particles[j].vel[0] = vx_i;
        }
        if vy_i.signum() == -vy_j.signum() {
            self.particles[i].vel[1] = vy_j;
            self.particles[j].vel[1] = vy_i;
        }
    }
    fn apply_force(&mut self, i: usize, j: usize) {
        let [xi, yi] = self.particles[i].pos;
        let [xj, yj] = self.particles[j].pos;
        let [dx, dy] = [xi - xj, yi - yj];
        let r = (dx * dx + dy * dy).sqrt();

        let [vxi, vyi] = &mut self.particles[i].vel;
        *vxi -= dx * self.simulation_dt * self.force / (r * r);
        *vyi -= dy * self.simulation_dt * self.force / (r * r);

        let [vxj, vyj] = &mut self.particles[j].vel;
        *vxj += dx * self.simulation_dt * self.force / (r * r);
        *vyj += dy * self.simulation_dt * self.force / (r * r);
    }
    fn heat(&mut self, i: usize) {
        let speed = self.particle_speed(i);
        let [vx, vy] = &mut self.particles[i].vel;
        let angle = vy.atan2(*vx);

        *vx = (speed + self.heating_rate) * angle.cos();
        *vy = (speed + self.heating_rate) * angle.sin();
    }
    fn cool(&mut self, i: usize) {
        let speed = self.particle_speed(i);
        let [vx, vy] = &mut self.particles[i].vel;
        let angle = vy.atan2(*vx);

        if speed <= self.heating_rate {
            *vx = 0.0;
            *vy = 0.0;
        } else {
            *vx = (speed - self.heating_rate) * angle.cos();
            *vy = (speed - self.heating_rate) * angle.sin();
        }
    }
    fn advance_particles(&mut self) {
        for p in &mut self.particles {
            let [vx, vy] = p.vel;

            let [dx, dy] = [vx * self.simulation_dt, vy * self.simulation_dt];
            let [x, y] = &mut p.pos;
            *x += dx;
            *y += dy;
        }
    }
    fn particle_speed(&self, i: usize) -> f64 {
        let p = self.particles[i];
        (p.vel[0] * p.vel[0] + p.vel[1] * p.vel[1]).sqrt()
    }
    fn detect_particle_collisions(&mut self) {
        for i in 0..self.particles.len() {
            for j in (i + 1)..self.particles.len() {
                // Particle collisions
                //
                let [xi, yi] = self.particles[i].pos;
                let [xj, yj] = self.particles[j].pos;
                let [dx, dy] = [xi - xj, yi - yj];
                let norm = (dx * dx + dy * dy).sqrt();
                if self.collisions.on && norm <= self.particle_r {
                    self.bounce(i, j);
                } else if self.vd_waals.on && norm <= self.force_r {
                    self.apply_force(i, j);
                }
            }
        }
    }
    fn detect_edge_collisions(&mut self) {
        for i in 0..self.particles.len() {
            let mut bounce_x = false;
            let mut bounce_y = false;
            let mut heat_x = 0;
            let [x, y] = &mut self.particles[i].pos;
            if *x <= 0.0 {
                *x = 0.0;
                bounce_x = true;
                heat_x = -1;
            }
            if *x >= SIMULATION_SIZE {
                *x = SIMULATION_SIZE;
                bounce_x = true;
                heat_x = 1;
            }
            if *y <= 0.0 {
                *y = 0.0;
                bounce_y = true;
            }
            if *y >= SIMULATION_SIZE {
                *y = SIMULATION_SIZE;
                bounce_y = true;
            }
            if bounce_x {
                self.bounce_x(i);
            }
            if bounce_y {
                self.bounce_y(i);
            }
            if self.heating.on {
                match heat_x {
                    -1 => self.cool(i),
                    1 => self.heat(i),
                    _ => {}
                }
            }
        }
    }
    fn evolve(&mut self) {
        self.advance_particles();
        if self.collisions.on || self.vd_waals.on {
            self.detect_particle_collisions();
        }
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
            cx.gfx.draw_shape(
                Shape::filled_circle(
                    Px::from_float(
                        (width + height) / 4.0
                            * (self.particle_r / SIMULATION_SIZE) as f32,
                    ),
                    Color::RED,
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
        self.draw_collision_btn(cx, &mut pos);
        pos += 10.0;
        self.draw_heating_btn(cx, &mut pos);
        pos += 10.0;
        self.draw_density_btn(cx, &mut pos);
        pos += 10.0;
        self.draw_temper_btn(cx, &mut pos);
        pos += 10.0;
        self.draw_vd_waals_btn(cx, &mut pos);
        pos += 10.0;
        self.draw_randomize_btn(cx, &mut pos);
    }
    fn draw_collision_btn(
        &mut self,
        cx: &mut GraphicsContext,
        x_pos: &mut f32,
    ) {
        let height = cx.gfx.size().height.into_float();
        let size = 150.0;
        let collisions_box = Rect::new(
            Point::px(*x_pos - 5.0, height - 9.0 / 10.0 * MARGIN),
            Size::px(size, MARGIN),
        );
        cx.gfx.draw_shape(&Shape::filled_rect(
            collisions_box,
            Color::new_f32(0.6, 0.6, 0.6, 1.0),
        ));
        cx.gfx.draw_text(
            Text::new(
                "Collisions",
                if self.collisions.on {
                    Color::GREEN
                } else {
                    Color::RED
                },
            )
            .translate_by(Point::px(*x_pos, height - 9.0 / 10.0 * MARGIN)),
        );
        *x_pos += size;
        if let Some(pos) = cx.cursor_position() {
            if collisions_box.contains(pos)
                && cx.mouse_button_pressed(MouseButton::Left)
            {
                if !self.collisions.guard {
                    self.collisions.on = !self.collisions.on;
                    self.collisions.guard = true;
                }
            } else {
                self.collisions.guard = false;
            }
        }
    }
    fn draw_heating_btn(&mut self, cx: &mut GraphicsContext, x_pos: &mut f32) {
        let size = 120.0;
        let height = cx.gfx.size().height.into_float();
        let heating_box = Rect::new(
            Point::px(*x_pos - 5.0, height - 9.0 / 10.0 * MARGIN),
            Size::px(size, MARGIN),
        );
        cx.gfx.draw_shape(&Shape::filled_rect(
            heating_box,
            Color::new_f32(0.6, 0.6, 0.6, 1.0),
        ));
        cx.gfx.draw_text(
            Text::new(
                "Heating",
                if self.heating.on {
                    Color::GREEN
                } else {
                    Color::RED
                },
            )
            .translate_by(Point::px(*x_pos, height - 9.0 / 10.0 * MARGIN)),
        );
        *x_pos += size;
        if let Some(pos) = cx.cursor_position() {
            if heating_box.contains(pos)
                && cx.mouse_button_pressed(MouseButton::Left)
            {
                if !self.heating.guard {
                    self.heating.on = !self.heating.on;
                    self.heating.guard = true;
                }
            } else {
                self.heating.guard = false;
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
    fn draw_vd_waals_btn(&mut self, cx: &mut GraphicsContext, x_pos: &mut f32) {
        let size = 80.0;
        let height = cx.gfx.size().height.into_float();
        let vd_waals_box = Rect::new(
            Point::px(*x_pos - 5.0, height - 9.0 / 10.0 * MARGIN),
            Size::px(size, MARGIN),
        );
        cx.gfx.draw_shape(&Shape::filled_rect(
            vd_waals_box,
            Color::new_f32(0.6, 0.6, 0.6, 1.0),
        ));
        cx.gfx.draw_text(
            Text::new(
                "vdW",
                if self.vd_waals.on {
                    Color::GREEN
                } else {
                    Color::RED
                },
            )
            .translate_by(Point::px(*x_pos, height - 9.0 / 10.0 * MARGIN)),
        );
        *x_pos += size;
        if let Some(pos) = cx.cursor_position() {
            if vd_waals_box.contains(pos)
                && cx.mouse_button_pressed(MouseButton::Left)
            {
                if !self.vd_waals.guard {
                    self.vd_waals.on = !self.vd_waals.on;
                    self.vd_waals.guard = true;
                }
            } else {
                self.vd_waals.guard = false;
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
        let temp_max = 1.5;
        for x in 0..DENSITY_PLY + 1 {
            for y in 0..DENSITY_PLY {
                let x_unit = width / DENSITY_PLY as f32;
                let y_unit = height / DENSITY_PLY as f32;

                let top_left = Point::px(x_unit * x as f32, y_unit * y as f32);

                let shade = temper[y][x] as f32 / temp_max as f32;
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
