use crate::components::keybindings::key;
use crate::repaint::{Cadence, Dirty};
use crate::theme::{self, lerp_u8};
use crate::update;
use ratatui::buffer::{Buffer, Cell};
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use std::cell::{Ref, RefCell};
use std::mem;
use std::ops::RangeInclusive;
use std::time::Instant;

const LOGO: &str = "maki";
const TAGLINE: &str = "the efficient coder";
const UPDATE_HINT: &str = " run maki update to get v";
const TIP_PREFIX: &str = "tip: ";
const HELP_SEGMENTS: &[(&str, bool)] = &[
    (key::HELP.label, true),
    (" help", false),
    (" · ", false),
    ("/help", true),
    (" in chat", false),
];

const TIPS: &[(&str, &str)] = &[
    (
        key::FILE_PICKER.label,
        "to grab file paths with fuzzy search",
    ),
    ("/tasks", "to see what your subagents are up to"),
    (key::SEARCH.label, "to find things in the conversation"),
    ("/btw", "to ask something without interrupting the session"),
    ("/memory", "to view, edit, and delete persistent notes"),
    ("/cd", "to switch to a different directory"),
];

const COLOR_TRANSITION_SECS: f32 = 0.4;

/// Seconds for the initial fade-in animation (ease-out cubic).
const FADE_DURATION: f32 = 1.6;
/// Seconds to wait before the logo starts appearing.
const LOGO_DELAY: f32 = 0.2;
/// Seconds over which the logo fades from dim to full brightness.
const LOGO_RAMP: f32 = 0.8;
/// How far accent-colored text is blended up from the background.
const ACCENT_ALPHA: f32 = 0.75;
/// The same, for secondary text that should recede.
const MUTED_ALPHA: f32 = 0.5;
const VERSION_ALPHA: f32 = 0.4;
/// Peak brightness of the logo, which never reaches the full accent color.
const LOGO_ALPHA: f32 = 0.85;
/// The logo's blue channel is nudged up so it reads slightly cooler than the
/// raw accent.
const LOGO_BLUE_LIFT: u8 = 15;
/// A soft highlight sweeps across the logo once per period.
const GLINT_PERIOD: f32 = 9.0;
const GLINT_START: f32 = 1.2;
const GLINT_SWEEP: f32 = 1.4;
/// Half-width of the highlight, in letters.
const GLINT_WIDTH: f32 = 1.6;
/// How far the highlight pulls the logo toward the foreground color.
const GLINT_STRENGTH: f32 = 0.35;

/// The field is a topographic map drawn in braille: each cell packs a 2×4 dot
/// grid, so contours resolve at eight samples per cell.
const DOTS_X: usize = 2;
const DOTS_Y: usize = 4;
const BRAILLE_BASE: u32 = 0x2800;
const BRAILLE_PATTERNS: usize = 256;
/// Every braille glyph is three bytes of UTF-8.
const BRAILLE_UTF8_LEN: usize = 3;
const BRAILLE_UTF8: [u8; BRAILLE_PATTERNS * BRAILLE_UTF8_LEN] = braille_utf8();
/// A ready-made cell per braille pattern, built at compile time. Their symbols
/// borrow static strings, so drawing one is a plain copy where
/// `Cell::set_char` would encode and copy a fresh string for every cell.
static BRAILLE_CELLS: [Cell; BRAILLE_PATTERNS] = braille_cells();
const BRAILLE_BITS: [[u8; DOTS_X]; DOTS_Y] =
    [[0x01, 0x08], [0x02, 0x10], [0x04, 0x20], [0x40, 0x80]];
const TOP_EDGE: usize = 0;
const RIGHT_EDGE: usize = 1;
const BOTTOM_EDGE: usize = 2;
const LEFT_EDGE: usize = 3;
/// Marching-squares cases come from four corner bits; a case and its
/// complement (`CASE_COUNT - 1 - case`) join the same edges, so the table only
/// covers the lower half.
const CASE_COUNT: usize = 16;
const CASE_FOLD: usize = CASE_COUNT / 2 - 1;
/// The edges a contour joins per folded case, corner bits clockwise from
/// top-left. Case 0 never reaches the table: a crossing level always has
/// corners on both sides.
const CASE_EDGES: [(usize, usize); CASE_FOLD + 1] = [
    (TOP_EDGE, TOP_EDGE),
    (TOP_EDGE, LEFT_EDGE),
    (TOP_EDGE, RIGHT_EDGE),
    (RIGHT_EDGE, LEFT_EDGE),
    (RIGHT_EDGE, BOTTOM_EDGE),
    (TOP_EDGE, RIGHT_EDGE),
    (TOP_EDGE, BOTTOM_EDGE),
    (BOTTOM_EDGE, LEFT_EDGE),
];
/// Opposite corners on the same side: the contour crosses all four edges.
const SADDLE_CASE: usize = 5;
const SADDLE_EDGES: (usize, usize) = (BOTTOM_EDGE, LEFT_EDGE);
/// Where each cell edge begins along the perimeter, clockwise from top-left.
const EDGE_STARTS: [f32; 4] = [
    0.0,
    DOTS_X as f32,
    (DOTS_X + DOTS_Y) as f32,
    (2 * DOTS_X + DOTS_Y) as f32,
];
const EDGE_LENGTHS: [f32; 4] = [DOTS_X as f32, DOTS_Y as f32, DOTS_X as f32, DOTS_Y as f32];
/// Contour endpoints snap to quarter dots along the cell perimeter.
const STEPS_PER_DOT: usize = 4;
const PERIMETER_STEPS: usize = 2 * (DOTS_X + DOTS_Y) * STEPS_PER_DOT;
/// Points sampled along each precomputed segment; dense enough that no
/// segment skips a dot it passes through.
const SEGMENT_SAMPLES: usize = 32;
/// A crossing at the very end of the left edge lands on step
/// [`PERIMETER_STEPS`], back at the top-left corner; giving it its own index
/// spares the hot path a modulo.
const PERIMETER_INDICES: usize = PERIMETER_STEPS + 1;
static SEGMENT_MASKS: [[u8; PERIMETER_INDICES]; PERIMETER_INDICES] = segment_masks();
/// World units per dot. Smaller = broader, calmer terrain.
const TERRAIN_SCALE: f32 = 0.05;
/// How strongly the terrain folds into itself.
const TERRAIN_WARP: f32 = 0.9;
const WARP_FREQS: (f32, f32) = (0.77, 0.9);
const WARP_SPEEDS: (f32, f32) = (-0.23, 0.31);
const TERRAIN_TERMS: usize = 4;
const TERM_FREQS: [f32; TERRAIN_TERMS] = [0.62, 0.71, 0.45, 0.38];
const TERM_SPEEDS: [f32; TERRAIN_TERMS] = [0.17, -0.13, -0.11, 0.07];
const TERM_WEIGHTS: [f32; TERRAIN_TERMS] = [1.0, 1.0, 0.55, 0.4];
/// Per corner column: summit dx², then sin and cos for every term.
const COLUMN_STREAMS: usize = 1 + 2 * TERRAIN_TERMS;
/// 1.5·2²³: adding and then subtracting it rounds an f32 to an integer.
const ROUND_MAGIC: f32 = 12_582_912.0;
/// The logo sits on a summit, so contours ring around it.
const SUMMIT_HEIGHT: f32 = 2.6;
const SUMMIT_RADIUS: f32 = 1.4;
/// Elevation between neighbouring contours. Larger = sparser lines.
const CONTOUR_SPACING: f32 = 0.42;
/// Contours per second that rise out of the summit and drift outward.
const CONTOUR_DRIFT: f32 = 0.18;
/// Every Nth contour is drawn brighter, like the index lines on a survey map.
const INDEX_CONTOUR_EVERY: i32 = 5;
const MINOR_CONTOUR_ALPHA: f32 = 0.5;
/// Cells of breathing room kept clear of contours around the text block.
const CLEARING_PAD: (f32, f32) = (6.0, 2.0);
/// How gradually contours return outside the clearing (squared-radius units).
const CLEARING_FEATHER: f32 = 2.4;
/// Contours fade out between these normalized distances from the center.
const EDGE_FADE: (f32, f32) = (0.45, 1.3);
/// Seconds for the map to unfold outward from the logo.
const REVEAL_DURATION: f32 = 2.4;
const REVEAL_FEATHER: f32 = 0.35;
/// Brightest contour as a blend from background toward the accent, per channel.
const FIELD_PEAK: [f32; 3] = [0.3, 0.21, 0.39];
const FIELD_SHADES: usize = 16;

const INV_TAU: f32 = 1.0 / std::f32::consts::TAU;
const TAU: f32 = std::f32::consts::TAU;
const PI: f32 = std::f32::consts::PI;
const FRAC_PI_2: f32 = std::f32::consts::FRAC_PI_2;
const BHASKARA_B: f32 = 4.0 / (PI * PI);

#[inline(always)]
fn fast_sin(x: f32) -> f32 {
    let x = x - fast_floor(x * INV_TAU) * TAU;
    let (x, sign) = if x > PI { (x - PI, -1.0_f32) } else { (x, 1.0) };
    let raw = BHASKARA_B * x * (PI - x);
    sign * (4.0 * raw) / (5.0 - raw)
}

/// `f32::floor` is a libm call on baseline x86-64, which keeps loops from
/// vectorizing; this stays in SSE2 registers. Exact for |x| < 2²².
#[inline(always)]
fn fast_floor(x: f32) -> f32 {
    let rounded = (x + ROUND_MAGIC) - ROUND_MAGIC;
    if rounded > x { rounded - 1.0 } else { rounded }
}

#[inline(always)]
fn fast_sincos(x: f32) -> (f32, f32) {
    (fast_sin(x), fast_sin(x + FRAC_PI_2))
}

/// Each terrain term is `sin(col_angle(x) + row_angle(y))` over a domain that
/// warps x by a function of y and y by a function of x. That keeps every term
/// separable, so a frame costs one sincos per column and per row, not per dot.
type TermFactors = [(f32, f32); TERRAIN_TERMS];

fn term_col_angles(x: f32, warp: f32) -> [f32; TERRAIN_TERMS] {
    [x, warp, x + warp, x - warp]
}

fn term_row_angles(y: f32, warp: f32) -> [f32; TERRAIN_TERMS] {
    [warp, y, y + warp, warp - y]
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if x <= edge0 {
        return 0.0;
    }
    if x >= edge1 {
        return 1.0;
    }
    let t = (x - edge0) / (edge1 - edge0);
    t * t * (3.0 - 2.0 * t)
}

/// Crossfading needs numbers to interpolate, so a palette color has nowhere to
/// go: it snaps and stays symbolic rather than being flattened to a guess at
/// what the terminal would have drawn.
pub struct ColorTransition {
    from: Color,
    to: Color,
    start: Instant,
}

impl ColorTransition {
    pub fn new(color: Color) -> Self {
        Self {
            from: color,
            to: color,
            start: Instant::now() - std::time::Duration::from_secs_f32(COLOR_TRANSITION_SECS),
        }
    }

    pub fn set(&mut self, color: Color) {
        if color == self.to {
            return;
        }
        let now = Instant::now();
        self.from = self.resolve_at(now);
        self.to = color;
        self.start = now;
    }

    /// A snapped transition has nothing left to draw, so it must not keep the
    /// repaint loop awake for the rest of its window.
    pub fn is_animating(&self) -> bool {
        matches!((self.from, self.to), (Color::Rgb(..), Color::Rgb(..)))
            && Instant::now().duration_since(self.start).as_secs_f32() < COLOR_TRANSITION_SECS
    }

    pub fn resolve(&self) -> Color {
        self.resolve_at(Instant::now())
    }

    fn resolve_at(&self, now: Instant) -> Color {
        let (Color::Rgb(fr, fg, fb), Color::Rgb(tr, tg, tb)) = (self.from, self.to) else {
            return self.to;
        };
        let t = (now.duration_since(self.start).as_secs_f32() / COLOR_TRANSITION_SECS).min(1.0);
        let p = ease_out_cubic(t);
        Color::Rgb(lerp_u8(fr, tr, p), lerp_u8(fg, tg, p), lerp_u8(fb, tb, p))
    }
}

/// The field's vignette depends only on the layout, so it is built once per
/// geometry rather than per frame, as parallel row-major per-cell arrays.
#[derive(Default)]
struct FieldMask {
    area: Rect,
    text_block: Rect,
    /// Where the summit sits, in cells from the area's origin.
    center: (f32, f32),
    /// Clearing around the text × fade toward the edges; zero where no
    /// contour may draw.
    alpha: Vec<f32>,
    /// Normalized distance from the center, swept by the entry reveal.
    radius: Vec<f32>,
}

impl FieldMask {
    fn new(area: Rect, text_block: Rect) -> Self {
        let (w, h) = (area.width as usize, area.height as usize);
        let center_x = (text_block.x - area.x) as f32 + text_block.width as f32 * 0.5;
        let center_y = (text_block.y - area.y) as f32 + text_block.height as f32 * 0.5;
        let inv_clearing_rx = 1.0 / (text_block.width as f32 * 0.5 + CLEARING_PAD.0);
        let inv_clearing_ry = 1.0 / (text_block.height as f32 * 0.5 + CLEARING_PAD.1);
        let (inv_half_w, inv_half_h) = (2.0 / w as f32, 2.0 / h as f32);

        let mut alpha = Vec::with_capacity(w * h);
        let mut radius = Vec::with_capacity(w * h);
        for row in 0..h {
            let dy = row as f32 + 0.5 - center_y;
            for col in 0..w {
                let dx = col as f32 + 0.5 - center_x;
                let clearing = (dx * inv_clearing_rx).powi(2) + (dy * inv_clearing_ry).powi(2);
                let distance = ((dx * inv_half_w).powi(2) + (dy * inv_half_h).powi(2)).sqrt();
                alpha.push(
                    smoothstep(1.0, CLEARING_FEATHER, clearing)
                        * (1.0 - smoothstep(EDGE_FADE.0, EDGE_FADE.1, distance)),
                );
                radius.push(distance);
            }
        }
        Self {
            area,
            text_block,
            center: (center_x, center_y),
            alpha,
            radius,
        }
    }

    fn fits(&self, area: Rect, text_block: Rect) -> bool {
        self.area == area && self.text_block == text_block
    }
}

pub struct Splash {
    start: Instant,
    field_offset: f32,
    animate: bool,
    tip_idx: usize,
    latest_version: Option<&'static str>,
    field_mask: RefCell<FieldMask>,
}

impl Default for Splash {
    fn default() -> Self {
        Self::new(true)
    }
}

impl Splash {
    pub fn new(animate: bool) -> Self {
        let mut rng = [0u8; 8];
        getrandom::fill(&mut rng).ok();
        let tip_idx = u32::from_le_bytes([rng[4], rng[5], rng[6], rng[7]]) as usize % TIPS.len();
        Self {
            start: Instant::now(),
            field_offset: (u64::from_le_bytes(rng) % 10_000) as f32,
            animate,
            tip_idx,
            latest_version: None,
            field_mask: RefCell::default(),
        }
    }

    /// The update check answers long after the splash is first painted, and
    /// storing its answer wakes nothing. Reading it in [`Self::render`] would
    /// put a version on screen that no poller ever saw, so a still splash
    /// (`splash_animation = false`) would never show the notice at all.
    pub fn poll_update(&mut self, latest: Option<&'static str>) -> Dirty {
        if self.latest_version == latest {
            return Dirty::NO;
        }
        self.latest_version = latest;
        Dirty::YES
    }

    /// The contour field drifts for as long as the splash is up. With it off the
    /// only motion left is the entry fade, which ends, so the loop settles on
    /// the start screen instead of burning a core on a still picture.
    pub fn cadence(&self) -> Cadence {
        Cadence::when(
            self.animate || self.start.elapsed().as_secs_f32() < FADE_DURATION,
            Cadence::SMOOTH,
        )
    }

    pub fn render(&self, area: Rect, buf: &mut Buffer, accent: Color) {
        if area.width < 20 || area.height < 5 {
            return;
        }

        let t = self.start.elapsed().as_secs_f32();
        let fade = if t >= FADE_DURATION {
            1.0
        } else {
            ease_out_cubic(t / FADE_DURATION)
        };

        let block_height = 8;
        let top_y = area.y + area.height.saturating_sub(block_height) / 2;
        let tag_y = top_y + 1;
        let help_y = tag_y + 2;
        let tip_y = help_y + 2;

        if self.animate {
            let text_width = self.tip_width().max(help_width());
            let text_block = Rect::new(
                area.x + area.width.saturating_sub(text_width) / 2,
                top_y,
                text_width.min(area.width),
                tip_y - top_y + 1,
            );
            self.render_field(area, buf, t, fade, accent, text_block);
        }
        self.render_logo(area, buf, t, fade, top_y, accent);
        render_centered_faded(area, buf, fade, 0.75, tag_y, TAGLINE);
        self.render_help(area, buf, fade, help_y, accent);
        self.render_tip(area, buf, fade, tip_y, accent);
        render_version(area, buf, fade, area.y, self.latest_version);
    }

    /// The field shades contours between the background and the accent, so it
    /// needs both as numbers. On a palette theme there is nothing to
    /// interpolate and the splash simply draws without it.
    /// The mask for this layout, rebuilt only when the layout changes.
    fn field_mask(&self, area: Rect, text_block: Rect) -> Ref<'_, FieldMask> {
        if !self.field_mask.borrow().fits(area, text_block) {
            self.field_mask.replace(FieldMask::new(area, text_block));
        }
        self.field_mask.borrow()
    }

    fn render_field(
        &self,
        area: Rect,
        buf: &mut Buffer,
        t: f32,
        fade: f32,
        accent: Color,
        text_block: Rect,
    ) {
        let theme = theme::current();
        let (Color::Rgb(ac_r, ac_g, ac_b), Color::Rgb(bg_r, bg_g, bg_b)) =
            (accent, theme.background)
        else {
            return;
        };
        if buf.area.intersection(area) != area {
            return;
        }
        let shades: [Color; FIELD_SHADES] = std::array::from_fn(|i| {
            let s = (i + 1) as f32 / FIELD_SHADES as f32;
            Color::Rgb(
                lerp_u8(bg_r, ac_r, s * FIELD_PEAK[0]),
                lerp_u8(bg_g, ac_g, s * FIELD_PEAK[1]),
                lerp_u8(bg_b, ac_b, s * FIELD_PEAK[2]),
            )
        });

        let mask = self.field_mask(area, text_block);

        let (w, h) = (area.width as usize, area.height as usize);
        let (center_x, center_y) = mask.center;
        let terrain_t = t + self.field_offset;
        // Wrapping on a whole index period keeps index contours from jumping.
        let drift = (terrain_t * CONTOUR_DRIFT).rem_euclid(INDEX_CONTOUR_EVERY as f32);
        let cell_span_x = DOTS_X as f32 * TERRAIN_SCALE;
        let cell_span_y = DOTS_Y as f32 * TERRAIN_SCALE;
        let inv_spacing = 1.0 / CONTOUR_SPACING;
        let corners_w = w + 1;

        // Terrain is sampled only at cell corners, already scaled into contour
        // levels; marching squares recovers sub-cell line positions from them.
        // Column factors are laid out as structure-of-arrays streams
        // (summit dx² | sin₀ | cos₀ | … | cos₃) so every corner row is one
        // fused pass over contiguous slices.
        let mut columns = vec![0.0_f32; corners_w * COLUMN_STREAMS];
        let (dx2_stream, factor_streams) = columns.split_at_mut(corners_w);
        for col in 0..corners_w {
            let x = (col as f32 - center_x) * cell_span_x;
            dx2_stream[col] = x * x;
            let warp = TERRAIN_WARP * fast_sin(WARP_FREQS.0 * x + WARP_SPEEDS.0 * terrain_t);
            let angles = term_col_angles(x, warp);
            for k in 0..TERRAIN_TERMS {
                let (s, c) = fast_sincos(TERM_FREQS[k] * angles[k]);
                let weight = TERM_WEIGHTS[k] * inv_spacing;
                factor_streams[2 * k * corners_w + col] = s * weight;
                factor_streams[(2 * k + 1) * corners_w + col] = c * weight;
            }
        }
        let [dx2, s0, c0, s1, c1, s2, c2, s3, c3]: [&[f32]; COLUMN_STREAMS] =
            std::array::from_fn(|i| &columns[i * corners_w..(i + 1) * corners_w]);

        let inv_summit_r2 = 1.0 / (SUMMIT_RADIUS * SUMMIT_RADIUS);
        let summit_levels = SUMMIT_HEIGHT * inv_spacing;
        let fill_row = |corner_row: usize, levels: &mut [f32], bands: &mut [f32]| {
            let y = (corner_row as f32 - center_y) * cell_span_y;
            let dy2 = y * y;
            let warp = TERRAIN_WARP * fast_sin(WARP_FREQS.1 * y + WARP_SPEEDS.1 * terrain_t);
            let angles = term_row_angles(y, warp);
            let [(rs0, rc0), (rs1, rc1), (rs2, rc2), (rs3, rc3)]: TermFactors =
                std::array::from_fn(|k| {
                    fast_sincos(TERM_FREQS[k] * angles[k] + TERM_SPEEDS[k] * terrain_t)
                });
            let (levels, bands) = (&mut levels[..corners_w], &mut bands[..corners_w]);
            // AUTOVECTORIZED: sin(a + b) = sin a·cos b + cos a·sin b per term.
            // Keep this loop free of branches, calls and non-contiguous reads.
            for col in 0..corners_w {
                let level = summit_levels / (1.0 + (dx2[col] + dy2) * inv_summit_r2)
                    + drift
                    + s0[col] * rc0
                    + c0[col] * rs0
                    + s1[col] * rc1
                    + c1[col] * rs1
                    + s2[col] * rc2
                    + c2[col] * rs2
                    + s3[col] * rc3
                    + c3[col] * rs3;
                levels[col] = level;
                bands[col] = fast_floor(level);
            }
        };

        let revealing = t < REVEAL_DURATION;
        let reveal = ease_out_cubic(t / REVEAL_DURATION) * (EDGE_FADE.1 + REVEAL_FEATHER);

        let mut level_rows = vec![0.0_f32; 4 * corners_w];
        let (level_rows, band_rows) = level_rows.split_at_mut(2 * corners_w);
        let (mut upper, mut lower) = level_rows.split_at_mut(corners_w);
        let (mut upper_bands, mut lower_bands) = band_rows.split_at_mut(corners_w);
        let mut crossed_cols = vec![0_u16; w];
        fill_row(0, upper, upper_bands);
        for row in 0..h {
            fill_row(row + 1, lower, lower_bands);
            let row_alpha = &mask.alpha[row * w..(row + 1) * w];
            let row_radius = &mask.radius[row * w..(row + 1) * w];
            let crossed =
                collect_crossed_cols(upper_bands, lower_bands, row_alpha, &mut crossed_cols);
            let row_start = buf.index_of(area.x, area.y + row as u16);
            let cells = &mut buf.content[row_start..row_start + w];
            for &col in &crossed_cols[..crossed] {
                let col = usize::from(col);
                let mut alpha = fade * row_alpha[col];
                if revealing {
                    alpha *= 1.0 - smoothstep(reveal - REVEAL_FEATHER, reveal, row_radius[col]);
                }

                let bands = [
                    upper_bands[col],
                    upper_bands[col + 1],
                    lower_bands[col + 1],
                    lower_bands[col],
                ];
                let corners = [upper[col], upper[col + 1], lower[col + 1], lower[col]];
                let (bits, on_index) = contour_cell(corners, crossing_levels(bands));
                if !on_index {
                    alpha *= MINOR_CONTOUR_ALPHA;
                }
                let shade = ((alpha * FIELD_SHADES as f32 + 0.5) as usize).min(FIELD_SHADES);
                if bits == 0 || shade == 0 {
                    continue;
                }
                let cell = &mut cells[col];
                let bg = cell.bg;
                cell.clone_from(&BRAILLE_CELLS[usize::from(bits)]);
                cell.set_fg(shades[shade - 1]).set_bg(bg);
            }
            mem::swap(&mut upper, &mut lower);
            mem::swap(&mut upper_bands, &mut lower_bands);
        }
    }

    fn render_logo(
        &self,
        area: Rect,
        buf: &mut Buffer,
        t: f32,
        fade: f32,
        top_y: u16,
        accent: Color,
    ) {
        let theme = theme::current();
        let bg = theme.background;

        let logo_x = area.x + (area.width.saturating_sub(LOGO.len() as u16)) / 2;
        let ramp = ease_out_cubic(((t - LOGO_DELAY) / LOGO_RAMP).clamp(0.0, 1.0)) * fade;
        let lifted = match accent {
            Color::Rgb(r, g, b) => Color::Rgb(r, g, b.saturating_add(LOGO_BLUE_LIFT)),
            symbolic => symbolic,
        };

        for (col, ch) in LOGO.chars().enumerate() {
            let x = logo_x + col as u16;
            if x >= area.x + area.width || top_y >= area.y + area.height {
                continue;
            }
            let shine = if self.animate { glint(t, col) } else { 0.0 };
            let color = blend(lifted, theme.foreground, shine * GLINT_STRENGTH).unwrap_or(lifted);
            let alpha = (LOGO_ALPHA + (1.0 - LOGO_ALPHA) * shine) * ramp;
            let style = faded_style(color, bg, alpha).add_modifier(Modifier::BOLD);
            if let Some(cell) = buf.cell_mut((x, top_y)) {
                cell.set_char(ch).set_style(style);
            }
        }
    }

    fn render_help(&self, area: Rect, buf: &mut Buffer, fade: f32, help_y: u16, accent: Color) {
        if help_y >= area.y + area.height {
            return;
        }

        let theme = theme::current();
        let bg = theme.background;
        let x_start = area.x + area.width.saturating_sub(help_width()) / 2;

        let segments: Vec<_> = HELP_SEGMENTS
            .iter()
            .map(|&(text, highlighted)| {
                let (target, alpha) = if highlighted {
                    (accent, ACCENT_ALPHA)
                } else {
                    (theme.foreground, MUTED_ALPHA)
                };
                (text, faded_style(target, bg, alpha * fade))
            })
            .collect();

        render_segments(area, buf, help_y, x_start, &segments);
    }

    fn tip_width(&self) -> u16 {
        let (label, desc) = TIPS[self.tip_idx];
        (TIP_PREFIX.len() + label.len() + 1 + desc.len()) as u16
    }

    fn render_tip(&self, area: Rect, buf: &mut Buffer, fade: f32, tip_y: u16, accent: Color) {
        if tip_y >= area.y + area.height {
            return;
        }

        let theme = theme::current();
        let bg = theme.background;
        let tip_fg = theme.todo_in_progress.fg.unwrap_or(Color::Yellow);

        let (label, desc) = TIPS[self.tip_idx];
        let x_start = area.x + area.width.saturating_sub(self.tip_width()) / 2;

        let segments: &[(&str, Style)] = &[
            (
                TIP_PREFIX,
                faded_style(tip_fg, bg, ACCENT_ALPHA * fade).add_modifier(Modifier::BOLD),
            ),
            (label, faded_style(accent, bg, ACCENT_ALPHA * fade)),
            (" ", Style::default()),
            (desc, faded_style(theme.foreground, bg, MUTED_ALPHA * fade)),
        ];

        render_segments(area, buf, tip_y, x_start, segments);
    }
}

fn help_width() -> u16 {
    HELP_SEGMENTS.iter().map(|(s, _)| s.len() as u16).sum()
}

/// Packs the columns of visible cells whose four corners straddle a contour band into
/// `crossed`, returning how many there are. Branch-free stream compaction:
/// every column is written and the cursor only advances on a crossing, so the
/// scalar marching-squares pass walks a dense list instead of mispredicting a
/// branch on every cell.
fn collect_crossed_cols(upper: &[f32], lower: &[f32], alpha: &[f32], crossed: &mut [u16]) -> usize {
    let cols = crossed.len();
    let (upper, lower, alpha) = (&upper[..=cols], &lower[..=cols], &alpha[..cols]);
    let mut count = 0;
    for col in 0..cols {
        let band = upper[col];
        let straddles = (band != upper[col + 1]) | (band != lower[col]) | (band != lower[col + 1]);
        crossed[count] = col as u16;
        count += usize::from(straddles & (alpha[col] > 0.0));
    }
    count
}

/// Marching squares over one cell: each contour level that crosses it enters
/// and leaves through the cell's perimeter, and the braille dots for that
/// segment come straight from [`SEGMENT_MASKS`]. `corners` run clockwise from
/// top-left, so corner `i` starts edge `i`. Edges come from a table rather
/// than a branch per case, which keeps the hot path free of jumps that
/// mispredict on every cell.
fn contour_cell(corners: [f32; 4], levels: RangeInclusive<i32>) -> (u8, bool) {
    let mut bits = 0;
    let mut on_index = false;
    for level in levels {
        let threshold = level as f32;
        let case = (corners[0] >= threshold) as usize
            | ((corners[1] >= threshold) as usize) << 1
            | ((corners[2] >= threshold) as usize) << 2
            | ((corners[3] >= threshold) as usize) << 3;
        let folded = if case > CASE_FOLD {
            CASE_COUNT - 1 - case
        } else {
            case
        };
        let (from, to) = CASE_EDGES[folded];
        bits |= segment_mask(&corners, threshold, from, to);
        if folded == SADDLE_CASE {
            let (from, to) = SADDLE_EDGES;
            bits |= segment_mask(&corners, threshold, from, to);
        }
        on_index |= level % INDEX_CONTOUR_EVERY == 0;
    }
    (bits, on_index)
}

/// The contour levels between a cell's lowest and highest corner band. Bands
/// are never NaN, so plain comparisons stand in for `f32::min`/`max` and their
/// NaN handling.
#[inline(always)]
fn crossing_levels([a, b, c, d]: [f32; 4]) -> RangeInclusive<i32> {
    let (low_ab, high_ab) = if a < b { (a, b) } else { (b, a) };
    let (low_cd, high_cd) = if c < d { (c, d) } else { (d, c) };
    let lowest = if low_ab < low_cd { low_ab } else { low_cd };
    let highest = if high_ab > high_cd { high_ab } else { high_cd };
    lowest as i32 + 1..=highest as i32
}

#[inline(always)]
fn segment_mask(corners: &[f32; 4], threshold: f32, from_edge: usize, to_edge: usize) -> u8 {
    SEGMENT_MASKS[perimeter_step(corners, threshold, from_edge)]
        [perimeter_step(corners, threshold, to_edge)]
}

/// Where the level crosses `edge`, in steps clockwise from the top-left corner.
#[inline(always)]
fn perimeter_step(corners: &[f32; 4], threshold: f32, edge: usize) -> usize {
    let (from, to) = (corners[edge], corners[(edge + 1) % 4]);
    let along = EDGE_STARTS[edge] + (threshold - from) / (to - from) * EDGE_LENGTHS[edge];
    (along * STEPS_PER_DOT as f32 + 0.5) as u8 as usize
}

const fn braille_utf8() -> [u8; BRAILLE_PATTERNS * BRAILLE_UTF8_LEN] {
    let mut bytes = [0; BRAILLE_PATTERNS * BRAILLE_UTF8_LEN];
    let mut pattern = 0;
    while pattern < BRAILLE_PATTERNS {
        let code = BRAILLE_BASE + pattern as u32;
        let at = pattern * BRAILLE_UTF8_LEN;
        bytes[at] = 0xE0 | (code >> 12) as u8;
        bytes[at + 1] = 0x80 | ((code >> 6) & 0x3F) as u8;
        bytes[at + 2] = 0x80 | (code & 0x3F) as u8;
        pattern += 1;
    }
    bytes
}

const fn braille_cells() -> [Cell; BRAILLE_PATTERNS] {
    let mut cells = [const { Cell::EMPTY }; BRAILLE_PATTERNS];
    let mut pattern = 0;
    while pattern < BRAILLE_PATTERNS {
        let (_, rest) = BRAILLE_UTF8.split_at(pattern * BRAILLE_UTF8_LEN);
        let (glyph, _) = rest.split_at(BRAILLE_UTF8_LEN);
        let Ok(glyph) = str::from_utf8(glyph) else {
            panic!("braille patterns are valid UTF-8");
        };
        // Const evaluation cannot run destructors; the empty cell being
        // replaced owns nothing.
        mem::forget(mem::replace(&mut cells[pattern], Cell::new(glyph)));
        pattern += 1;
    }
    cells
}

/// Walks the cell perimeter clockwise from the top-left corner, in dot units.
/// Crossings round to the nearest step, so opposite edges quantize alike and a
/// straight contour stays straight.
const fn perimeter_point(step: usize) -> (f32, f32) {
    let along = (step % PERIMETER_STEPS) as f32 / STEPS_PER_DOT as f32;
    let (w, h) = (DOTS_X as f32, DOTS_Y as f32);
    if along < w {
        (along, 0.0)
    } else if along < w + h {
        (w, along - w)
    } else if along < 2.0 * w + h {
        (2.0 * w + h - along, h)
    } else {
        (0.0, 2.0 * (w + h) - along)
    }
}

const fn clamp_dot(position: f32, dots: usize) -> usize {
    let dot = position as usize;
    if dot >= dots { dots - 1 } else { dot }
}

/// Braille dots covered by the straight segment between every pair of
/// perimeter steps, so rasterizing a contour costs a lookup per cell.
const fn segment_masks() -> [[u8; PERIMETER_INDICES]; PERIMETER_INDICES] {
    let mut masks = [[0; PERIMETER_INDICES]; PERIMETER_INDICES];
    let mut from = 0;
    while from < PERIMETER_INDICES {
        let (x0, y0) = perimeter_point(from);
        let mut to = 0;
        while to < PERIMETER_INDICES {
            let (x1, y1) = perimeter_point(to);
            let mut sample = 0;
            while sample <= SEGMENT_SAMPLES {
                let t = sample as f32 / SEGMENT_SAMPLES as f32;
                let dot_x = clamp_dot(x0 + (x1 - x0) * t, DOTS_X);
                let dot_y = clamp_dot(y0 + (y1 - y0) * t, DOTS_Y);
                masks[from][to] |= BRAILLE_BITS[dot_y][dot_x];
                sample += 1;
            }
            to += 1;
        }
        from += 1;
    }
    masks
}

/// Brightness of the highlight over a logo letter: a smooth bump that sweeps
/// left to right once per [`GLINT_PERIOD`].
fn glint(t: f32, letter: usize) -> f32 {
    if t < GLINT_START {
        return 0.0;
    }
    let progress = ((t - GLINT_START) % GLINT_PERIOD) / GLINT_SWEEP;
    let center = progress * (LOGO.len() as f32 + 2.0 * GLINT_WIDTH) - GLINT_WIDTH;
    let d = (letter as f32 - center) / GLINT_WIDTH;
    let falloff = (1.0 - d * d).max(0.0);
    falloff * falloff
}

fn render_version(area: Rect, buf: &mut Buffer, fade: f32, y: u16, new_version: Option<&str>) {
    if y >= area.y + area.height {
        return;
    }
    let theme = theme::current();
    let bg = theme.background;
    let text = match new_version {
        Some(v) => format!("v{}{UPDATE_HINT}{v}", update::CURRENT),
        None => format!("v{}", update::CURRENT),
    };
    let style = faded_style(theme.foreground, bg, VERSION_ALPHA * fade);
    let x_start = area.x + area.width.saturating_sub(text.chars().count() as u16 + 1);
    render_segments(area, buf, y, x_start, &[(&text, style)]);
}

fn render_centered_faded(
    area: Rect,
    buf: &mut Buffer,
    fade: f32,
    intensity: f32,
    y: u16,
    text: &str,
) {
    if y >= area.y + area.height {
        return;
    }
    let theme = theme::current();
    let bg = theme.background;
    let style = faded_style(theme.foreground, bg, intensity * fade);
    let x_start = area.x + area.width.saturating_sub(text.chars().count() as u16) / 2;
    render_segments(area, buf, y, x_start, &[(text, style)]);
}

/// Fading means blending toward the background, which needs both ends as
/// numbers. A palette color has no value to blend, so it is drawn flat instead
/// of being resolved to a guess at what the terminal would have shown.
fn faded_style(fg: Color, bg: Color, alpha: f32) -> Style {
    Style::new().bg(bg).fg(blend(bg, fg, alpha).unwrap_or(fg))
}

fn blend(from: Color, to: Color, t: f32) -> Option<Color> {
    match (from, to) {
        (Color::Rgb(fr, fg, fb), Color::Rgb(tr, tg, tb)) => Some(Color::Rgb(
            lerp_u8(fr, tr, t),
            lerp_u8(fg, tg, t),
            lerp_u8(fb, tb, t),
        )),
        _ => None,
    }
}

fn render_segments(area: Rect, buf: &mut Buffer, y: u16, x_start: u16, segments: &[(&str, Style)]) {
    let x_end = area.x + area.width;
    let mut x = x_start;
    for &(text, style) in segments {
        for ch in text.chars() {
            if x >= x_end {
                return;
            }
            if let Some(cell) = buf.cell_mut((x, y)) {
                cell.set_char(ch).set_style(style);
            }
            x += 1;
        }
    }
}

fn ease_out_cubic(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    1.0 - (1.0 - t).powi(3)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::buffer_text;
    use crate::repaint::expect::{OWED, QUIET};
    use std::time::Duration;
    use test_case::test_case;

    fn transition_at(from: (u8, u8, u8), to: (u8, u8, u8), offset: Duration) -> Color {
        let mut ct = ColorTransition::new(Color::Rgb(from.0, from.1, from.2));
        ct.set(Color::Rgb(to.0, to.1, to.2));
        ct.resolve_at(ct.start + offset)
    }

    #[test]
    fn interpolation_over_time() {
        let start = transition_at((0, 0, 0), (200, 200, 200), Duration::ZERO);
        assert_eq!(start, Color::Rgb(0, 0, 0));

        let Color::Rgb(mid, _, _) =
            transition_at((0, 0, 0), (200, 200, 200), Duration::from_millis(200))
        else {
            panic!("an rgb transition stays rgb");
        };
        assert!(mid > 0 && mid < 200, "expected interpolated, got {mid}");

        let done = transition_at((0, 0, 0), (255, 255, 255), Duration::from_millis(500));
        assert_eq!(done, Color::Rgb(255, 255, 255));
    }

    #[test]
    fn chained_set_restarts_toward_new_target() {
        let mut ct = ColorTransition::new(Color::Rgb(0, 0, 0));
        ct.set(Color::Rgb(200, 100, 50));
        ct.set(Color::Rgb(10, 20, 30));

        let done = ct.resolve_at(ct.start + Duration::from_secs(1));
        assert_eq!(done, Color::Rgb(10, 20, 30));
    }

    const NEW_VERSION: &str = "99.9.9";
    const UNPOLLED: &str = "an unpolled version must not appear on screen";
    const POLLED: &str = "a polled version must appear on screen";

    fn rendered(splash: &Splash) -> String {
        let area = Rect::new(0, 0, 80, 20);
        let mut buf = Buffer::empty(area);
        splash.render(area, &mut buf, Color::Blue);
        buffer_text(&buf)
    }

    /// The check answers on its own, so the notice only reaches the screen
    /// because a poll reported it. A still splash owes no other frame, so
    /// reading the answer in `render` would hide it until the user typed.
    #[test]
    fn an_update_notice_reaches_the_screen_only_after_a_poll() {
        let mut splash = Splash::new(false);
        assert_eq!(splash.poll_update(None), Dirty::NO, "{QUIET}");
        assert!(!rendered(&splash).contains(NEW_VERSION), "{UNPOLLED}");

        assert_eq!(splash.poll_update(Some(NEW_VERSION)), Dirty::YES, "{OWED}");
        assert_eq!(splash.poll_update(Some(NEW_VERSION)), Dirty::NO, "{QUIET}");

        let screen = rendered(&splash);
        assert!(screen.contains(UPDATE_HINT.trim_end()), "{POLLED}");
        assert!(screen.contains(NEW_VERSION), "{POLLED}");
    }

    /// `splash_animation = false` is what a user on a slow machine reaches
    /// for, so the start screen really has to stop painting once the entry
    /// fade is over.
    #[test_case(false, false => Cadence::SMOOTH ; "entry_fade_is_running")]
    #[test_case(false, true  => Cadence::IDLE   ; "still_splash_settles_after_the_fade")]
    #[test_case(true,  true  => Cadence::SMOOTH ; "contour_field_drifts_for_as_long_as_it_is_up")]
    fn splash_cadence(animate: bool, faded: bool) -> Cadence {
        let mut splash = Splash::new(animate);
        if faded {
            splash.start -= Duration::from_secs_f32(FADE_DURATION);
        }
        splash.cadence()
    }

    #[test]
    fn is_animating_lifecycle() {
        let ct = ColorTransition::new(Color::Rgb(0, 0, 0));
        assert!(!ct.is_animating(), "settled on construction");

        let mut ct = ColorTransition::new(Color::Rgb(0, 0, 0));
        ct.set(Color::Rgb(255, 0, 0));
        assert!(ct.is_animating(), "animating after set");
    }

    /// Resolving a palette color to rgb here would defeat the point of asking
    /// for it, so it crosses the transition untouched.
    #[test_case(Color::Blue, Color::Rgb(1, 2, 3); "into rgb")]
    #[test_case(Color::Rgb(1, 2, 3), Color::Blue; "out of rgb")]
    fn a_palette_color_snaps_instead_of_blending(from: Color, to: Color) {
        let mut ct = ColorTransition::new(from);
        ct.set(to);
        assert_eq!(ct.resolve_at(ct.start), to);
        assert!(
            !ct.is_animating(),
            "a snapped transition must not keep asking for frames"
        );
    }

    const ANSI_THEME_NAME: &str = "an all-ansi theme must parse";
    const ANSI_THEME: &str = r#"
[palette]
background = "black"
foreground = "white"

[ui]
background = { bg = "background" }
foreground = { fg = "foreground" }
todo_in_progress = { fg = "yellow" }
"#;

    /// The contour field needs numbers to shade its lines; on a palette theme
    /// the splash draws flat rather than inventing truecolor for it. The theme
    /// is installed here rather than inherited, because the default one is
    /// truecolor and would make this pass or fail on test ordering.
    #[test]
    fn a_palette_accent_keeps_the_splash_free_of_truecolor() {
        theme::set(theme::Theme::from_toml(ANSI_THEME).expect(ANSI_THEME_NAME));

        let mut splash = Splash::new(true);
        // At t=0 the entry fade has not started painting, so the assertion
        // below would hold no matter what the field did.
        splash.start -= Duration::from_secs_f32(FADE_DURATION);
        let area = Rect::new(0, 0, 80, 20);
        let mut buf = Buffer::empty(area);
        splash.render(area, &mut buf, Color::Blue);

        let rgb = buf
            .content
            .iter()
            .filter_map(|c| c.style().fg)
            .find(|fg| matches!(fg, Color::Rgb(..)));
        assert_eq!(rgb, None, "a palette theme must not produce rgb cells");
    }

    /// The magic-number floor must agree with `f32::floor` on both sides of
    /// zero and right at integers, where a band boundary sits.
    #[test_case(-2.5)]
    #[test_case(-1.0)]
    #[test_case(-0.25)]
    #[test_case(0.0)]
    #[test_case(0.5)]
    #[test_case(0.999)]
    #[test_case(1.0)]
    #[test_case(1234.75)]
    #[test_case(-1234.25)]
    fn fast_floor_matches_floor(x: f32) {
        assert_eq!(fast_floor(x), x.floor());
    }

    #[test]
    fn braille_cells_hold_their_own_pattern() {
        for (pattern, cell) in BRAILLE_CELLS.iter().enumerate() {
            let expected = char::from_u32(BRAILLE_BASE + pattern as u32).map(String::from);
            assert_eq!(
                Some(cell.symbol().to_owned()),
                expected,
                "pattern {pattern:#04x}"
            );
        }
    }

    const ROW_BELOW_MIDDLE: usize = DOTS_Y / 2;
    const RIGHT_COLUMN: usize = DOTS_X - 1;

    /// A level crossing a cell straight through its middle must come out as a
    /// straight line of dots, so the case table joins the right edges and
    /// opposite edges quantize alike. Corners run clockwise from top-left.
    #[test_case([0.0, 0.0, 2.0, 2.0], BRAILLE_BITS[ROW_BELOW_MIDDLE][0] | BRAILLE_BITS[ROW_BELOW_MIDDLE][1] ; "horizontal")]
    #[test_case([0.0, 2.0, 2.0, 0.0], BRAILLE_BITS.iter().fold(0, |bits, row| bits | row[RIGHT_COLUMN]) ; "vertical")]
    fn a_level_through_the_middle_draws_a_straight_line(corners: [f32; 4], expected: u8) {
        let (bits, _) = contour_cell(corners, 1..=1);
        assert_eq!(bits, expected);
    }

    /// The mask is cached per layout; a resize that kept the old one would
    /// index past its rows.
    #[test]
    fn a_resize_rebuilds_the_field_mask() {
        let splash = Splash::new(true);
        let text_block = Rect::new(10, 5, 40, 5);
        for area in [Rect::new(0, 0, 80, 20), Rect::new(0, 0, 120, 40)] {
            let mask = splash.field_mask(area, text_block);
            assert_eq!(mask.alpha.len(), area.area() as usize);
            assert_eq!(mask.radius.len(), area.area() as usize);
        }
    }
}
