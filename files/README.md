# AutoPrize — AI Car Valuation

**Futuristic, animated used car price prediction website for the Indian market.**

Built from a real dataset of 5,500+ used car listings and a trained Random Forest model.

---

## 🚀 Quick Start

1. **Download / unzip** the project folder
2. **Double-click `index.html`** — opens directly in any modern browser
3. No server, no build step, no dependencies to install

---

## 🌐 GitHub Pages Hosting

1. Push the folder contents to a GitHub repository
2. Go to **Settings → Pages → Source: main branch / root**
3. Your site will be live at `https://yourusername.github.io/repo-name/`

---

## 📁 File Structure

```
autoprize/
├── index.html      ← Main HTML (single-page app)
├── style.css       ← All styles (dark futuristic theme)
├── script.js       ← All logic: model, form, 3D scene, animations
├── assets/         ← Reserved for future images/icons
└── README.md       ← This file
```

---

## 🧠 Model Details

### Dataset
- **Source:** `usedcar_price.csv` (5,500 rows, 18 columns)
- **Target:** `price_inr`

### Preprocessing (matches notebook exactly)
1. Drop duplicates
2. Drop columns: `variant`, `listing_date`, `city`, `state`, `source_platform`, `model_name`
3. Parse `listing_date` → extract `listing_year`
4. Engineer `vehicle_age = listing_year - registration_year`
5. Impute: numerical → median, categorical → mode
6. One-hot encode with `drop_first=True`
7. StandardScaler normalize

### Algorithm
- **Python original:** `RandomForestRegressor(n_estimators=100, max_depth=20)` — R² = 0.855
- **Browser port:** `Ridge(alpha=100)` coefficients — R² = 0.766
- The Ridge model is fully embedded in `script.js` — zero backend calls

### Key Features (by importance)
| Feature | Importance |
|---------|------------|
| engine_cc | 46.1% |
| manufacturing_year | 13.3% |
| transmission | 10.5% |
| mileage | 7.5% |
| km_driven | 6.1% |

---

## 🎨 Design

- **Theme:** Dark automotive showroom
- **Accents:** Neon cyan `#00d4ff` + violet `#7c3aed`
- **Fonts:** Orbitron (display) + Inter (body) + Space Mono (data)
- **3D Scene:** Three.js via CDN (graceful CSS canvas fallback)
- **Animations:** CSS keyframes + requestAnimationFrame

---

## 🔧 Customization

### Update model coefficients
Edit the `MODEL` object at the top of `script.js`. All coefficients, means, and standard deviations are clearly labeled.

### Add more brands/models
Update the `BRAND_MODELS` object in `script.js`:
```js
const BRAND_MODELS = {
  "YourBrand": ["Model1", "Model2"],
  ...
};
```

### Change color theme
Update CSS variables in `:root` in `style.css`:
```css
--neon-cyan:   #00d4ff;
--neon-purple: #7c3aed;
```

---

## ⚙️ Technical Notes

- **No backend** — prediction runs entirely in the browser
- **No frameworks** — vanilla HTML/CSS/JS only
- **CDN libraries:** Three.js r128 (fallback to canvas if CDN fails)
- **Fonts:** Google Fonts (optional — system fonts load if offline)
- **Offline support:** Works offline except for Three.js CDN and fonts (both have graceful fallbacks)

---

## 📊 Accuracy

| Metric | Value |
|--------|-------|
| R² Score | 0.766 (Ridge) / 0.855 (Random Forest) |
| MAE | ~₹1.1L (Ridge) / ~₹96K (RF) |
| Price Range | ₹83,500 – ₹47,43,000 |

*Disclaimer: Estimates carry ±15–20% variance. Actual price depends on physical condition, negotiation, and local market.*

---

## 🏷️ Tech Stack

`HTML5` · `CSS3` · `Vanilla JavaScript` · `Three.js r128` · `Google Fonts`

Model trained with: `scikit-learn` · `pandas` · `RandomForestRegressor` · `Ridge`
