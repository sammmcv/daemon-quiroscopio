use crate::types::SampleFrame;

#[derive(Clone, Copy, Debug)]
pub struct Quaternion {
    pub w: f32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Quaternion {
    pub fn new(w: f32, x: f32, y: f32, z: f32) -> Self {
        Self { w, x, y, z }
    }

    pub fn from_sample(frame: &SampleFrame, sensor_idx: usize) -> Self {
        let qw = frame.qw[sensor_idx];
        let qx = frame.qx[sensor_idx];
        let qy = frame.qy[sensor_idx];
        let qz = frame.qz[sensor_idx];
        Self::new(qw, qx, qy, qz).normalized()
    }

    pub fn normalized(self) -> Self {
        let norm = (self.w * self.w
            + self.x * self.x
            + self.y * self.y
            + self.z * self.z)
            .sqrt()
            .max(1e-9);

        let mut q = Self {
            w: self.w / norm,
            x: self.x / norm,
            y: self.y / norm,
            z: self.z / norm,
        };

        // Force a canonical representation (w >= 0) to avoid sudden flips between q and -q
        if q.w < 0.0 {
            q.w = -q.w;
            q.x = -q.x;
            q.y = -q.y;
            q.z = -q.z;
        }

        q
    }

    pub fn conjugate(self) -> Self {
        Self {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    pub fn mul(self, rhs: Self) -> Self {
        Self {
            w: self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
            x: self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y,
            y: self.w * rhs.y - self.x * rhs.z + self.y * rhs.w + self.z * rhs.x,
            z: self.w * rhs.z + self.x * rhs.y - self.y * rhs.x + self.z * rhs.w,
        }
    }

    /// Aproxima la velocidad angular que transforma `self` en `other` en un intervalo `dt`.
    pub fn angular_velocity_to(self, other: Self, dt: f32) -> (f32, f32, f32) {
        if dt <= 1e-6 {
            return (0.0, 0.0, 0.0);
        }

        let delta = self.conjugate().mul(other).normalized();
        let w = delta.w.clamp(-1.0, 1.0);
        let angle = 2.0 * w.acos();

        if angle.abs() < 1e-6 {
            return (
                2.0 * delta.x / dt,
                2.0 * delta.y / dt,
                2.0 * delta.z / dt,
            );
        }

        let sin_half_sq = (1.0 - w * w).max(0.0);
        let sin_half = sin_half_sq.sqrt().max(1e-9);
        let axis_x = delta.x / sin_half;
        let axis_y = delta.y / sin_half;
        let axis_z = delta.z / sin_half;

        (
            axis_x * angle / dt,
            axis_y * angle / dt,
            axis_z * angle / dt,
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GyroMouseConfig {
    pub deadzone_omega: f32,
    pub gain_x: f32,
    pub gain_y: f32,
    pub max_speed: f32,
    pub alpha: f32,
    pub axis_sign_x: f32,
    pub axis_sign_y: f32,
    pub horizontal_axis: MotionAxis,
    pub vertical_axis: MotionAxis,
}

#[derive(Clone, Copy, Debug)]
pub enum MotionAxis {
    Rx,
    Ry,
    Rz,
}

impl MotionAxis {
    fn sample(self, rx: f32, ry: f32, rz: f32) -> f32 {
        match self {
            MotionAxis::Rx => rx,
            MotionAxis::Ry => ry,
            MotionAxis::Rz => rz,
        }
    }
}

impl Default for GyroMouseConfig {
    fn default() -> Self {
        Self {
            deadzone_omega: 0.045, // rad/s (3x mayor)
            gain_x: 500.0,         // px per rad/s
            gain_y: 500.0,
            max_speed: 800.0,
            alpha: 0.55,
            axis_sign_x: 1.0,
            axis_sign_y: 1.0,
            horizontal_axis: MotionAxis::Rx, // pitch controla X
            vertical_axis: MotionAxis::Ry,   // yaw controla Y
        }
    }
}

pub struct GyroMouseFilter {
    prev_dx: f32,
    prev_dy: f32,
    config: GyroMouseConfig,
}

impl GyroMouseFilter {
    pub fn new(config: GyroMouseConfig) -> Self {
        Self {
            prev_dx: 0.0,
            prev_dy: 0.0,
            config,
        }
    }

    pub fn reset(&mut self) {
        self.prev_dx = 0.0;
        self.prev_dy = 0.0;
    }

    /// `wx`, `wy`, `wz` son velocidades angulares (rad/s) ya estimadas.
    pub fn update(&mut self, wx: f32, wy: f32, wz: f32) -> (i32, i32) {
        let rx = wx;
        let ry = wy;
        let rz = wz;

        let mut vx = self.config.horizontal_axis.sample(rx, ry, rz);
        let mut vy = self.config.vertical_axis.sample(rx, ry, rz);

        if vx.abs() < self.config.deadzone_omega {
            vx = 0.0;
        }
        if vy.abs() < self.config.deadzone_omega {
            vy = 0.0;
        }

        if vx == 0.0 && vy == 0.0 {
            self.prev_dx *= 1.0 - self.config.alpha;
            self.prev_dy *= 1.0 - self.config.alpha;
            return (self.prev_dx.round() as i32, self.prev_dy.round() as i32);
        }

        let mut dx = vx * self.config.axis_sign_x * self.config.gain_x;
        let mut dy = vy * self.config.axis_sign_y * self.config.gain_y;

        dx = dx.clamp(-self.config.max_speed, self.config.max_speed);
        dy = dy.clamp(-self.config.max_speed, self.config.max_speed);

        let filtered_dx = self.config.alpha * dx + (1.0 - self.config.alpha) * self.prev_dx;
        let filtered_dy = self.config.alpha * dy + (1.0 - self.config.alpha) * self.prev_dy;

        self.prev_dx = filtered_dx;
        self.prev_dy = filtered_dy;

        (filtered_dx.round() as i32, filtered_dy.round() as i32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_4;

    #[test]
    fn deadzone_blocks_small_motion() {
        let mut filter = GyroMouseFilter::new(GyroMouseConfig::default());
        let (dx, dy) = filter.update(0.0, 0.01, 0.0);
        assert_eq!((dx, dy), (0, 0));
    }

    #[test]
    fn axis_sign_inverts_direction() {
        let mut filter_pos = GyroMouseFilter::new(GyroMouseConfig::default());
        let cfg_negative = GyroMouseConfig {
            axis_sign_x: -1.0,
            ..GyroMouseConfig::default()
        };
        let mut filter_neg = GyroMouseFilter::new(cfg_negative);

        let (dx_pos, _) = filter_pos.update(1.5, 0.0, 0.0);
        let (dx_neg, _) = filter_neg.update(1.5, 0.0, 0.0);

        assert_ne!(dx_pos, 0);
        assert_eq!(dx_pos, -dx_neg);
    }

    #[test]
    fn axis_mapping_can_swap_components() {
        let cfg_swapped = GyroMouseConfig {
            horizontal_axis: MotionAxis::Rz,
            vertical_axis: MotionAxis::Ry,
            axis_sign_y: 1.0,
            ..GyroMouseConfig::default()
        };
        let mut filter = GyroMouseFilter::new(cfg_swapped);

        let (dx_z, dy_z) = filter.update(0.0, 0.0, 3.0);
        assert_ne!(dx_z, 0);
        assert_eq!(dy_z, 0);

        filter.reset();
        let (_dx_y, dy_y) = filter.update(0.0, 3.0, 0.0);
        assert_ne!(dy_y, 0);
    }

    #[test]
    fn quaternion_to_angular_velocity_matches_rotation() {
        let prev = Quaternion::new(1.0, 0.0, 0.0, 0.0);
        let half = FRAC_PI_4 / 2.0;
        let current = Quaternion::new(half.cos(), 0.0, half.sin(), 0.0).normalized();
        let (wx, wy, wz) = prev.angular_velocity_to(current, 0.02);
        assert!(wx.abs() < 1e-3);
        assert!(wz.abs() < 1e-3);
        let expected = FRAC_PI_4 / 0.02;
        assert!((wy - expected).abs() < expected * 0.05);
    }
}
