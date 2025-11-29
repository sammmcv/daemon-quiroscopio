use std::time::Duration;
use uinput::device::Device;
use uinput::event::controller;
use uinput::event::keyboard;
use uinput::event::relative;

#[derive(Debug, Clone, Copy)]
pub enum GestureAction {
    SlideDer,
    SlideIzq,
    ZoomIn,
    ZoomOut,
    Grab,
    Drop,
}

pub struct HidOutput {
    dev: Device,
}

impl HidOutput {
    pub fn new() -> Result<Self, uinput::Error> {
        let dev = uinput::default()?
            .name("ble-gesture-hid")?
            .event(uinput::event::Keyboard::All)?
            .event(uinput::event::Controller::Mouse(controller::Mouse::Left))?
            .event(uinput::event::Relative::Position(relative::Position::X))?
            .event(uinput::event::Relative::Position(relative::Position::Y))?
            .create()?;

        Ok(HidOutput { dev })
    }

    fn sync(&mut self) -> Result<(), uinput::Error> {
        self.dev.synchronize()
    }

    fn key_tap(&mut self, key: keyboard::Key) -> Result<(), uinput::Error> {
        self.dev.press(&keyboard::Keyboard::Key(key))?;
        self.sync()?;
        std::thread::sleep(Duration::from_millis(10));
        self.dev.release(&keyboard::Keyboard::Key(key))?;
        self.sync()
    }

    fn ctrl_combo(&mut self, key: keyboard::Key) -> Result<(), uinput::Error> {
        self.dev
            .press(&keyboard::Keyboard::Key(keyboard::Key::LeftControl))?;
        self.sync()?;
        std::thread::sleep(Duration::from_millis(10));
        self.dev.press(&keyboard::Keyboard::Key(key))?;
        self.sync()?;
        std::thread::sleep(Duration::from_millis(10));
        self.dev.release(&keyboard::Keyboard::Key(key))?;
        self.sync()?;
        self.dev
            .release(&keyboard::Keyboard::Key(keyboard::Key::LeftControl))?;
        self.sync()
    }

    fn slide_der(&mut self) -> Result<(), uinput::Error> {
        self.key_tap(keyboard::Key::Right)
    }

    fn slide_izq(&mut self) -> Result<(), uinput::Error> {
        self.key_tap(keyboard::Key::Left)
    }

    fn zoom_in(&mut self) -> Result<(), uinput::Error> {
        self.ctrl_combo(keyboard::Key::Equal)
    }

    fn zoom_out(&mut self) -> Result<(), uinput::Error> {
        self.ctrl_combo(keyboard::Key::Minus)
    }

    fn grab(&mut self) -> Result<(), uinput::Error> {
        self.dev
            .press(&controller::Controller::Mouse(controller::Mouse::Left))?;
        self.sync()
    }

    fn drop(&mut self) -> Result<(), uinput::Error> {
        self.dev
            .release(&controller::Controller::Mouse(controller::Mouse::Left))?;
        self.sync()
    }

    /// Hace un click simple (press + release)
    fn click_left(&mut self) -> Result<(), uinput::Error> {
        self.dev
            .press(&controller::Controller::Mouse(controller::Mouse::Left))?;
        self.sync()?;
        std::thread::sleep(Duration::from_millis(10));
        self.dev
            .release(&controller::Controller::Mouse(controller::Mouse::Left))?;
        self.sync()
    }

    /// Mueve el cursor en relación a su posición actual (dx, dy en píxeles)
    pub fn move_cursor(&mut self, dx: i32, dy: i32) -> Result<(), uinput::Error> {
        self.dev.send(relative::Position::X, dx)?;
        self.dev.send(relative::Position::Y, dy)?;
        self.sync()
    }

    pub fn send(&mut self, action: GestureAction) -> Result<(), uinput::Error> {
        self.send_with_mode(action, false)
    }

    pub fn send_with_mode(&mut self, action: GestureAction, is_cursor_mode: bool) -> Result<(), uinput::Error> {
        match action {
            GestureAction::SlideDer => self.slide_der(),
            GestureAction::SlideIzq => self.slide_izq(),
            GestureAction::ZoomIn => {
                if is_cursor_mode {
                    self.click_left()
                } else {
                    self.zoom_in()
                }
            }
            GestureAction::ZoomOut => self.zoom_out(),
            GestureAction::Grab => self.grab(),
            GestureAction::Drop => self.drop(),
        }
    }
}
