/**
 * AutoPrize — AI Car Price Prediction
 * ============================================================
 * This file handles:
 *  1. Background particle canvas animation
 *  2. Three.js 3D car scene (hero)
 *  3. Dataset (brands/models from CSV analysis)
 *  4. Form population & dependent dropdowns
 *  5. Form validation
 *  6. Prediction engine (Ridge regression model ported from Python)
 *  7. Result rendering
 * ============================================================
 */

/* ============================================================
   1. DATASET — Brand → Model mapping extracted from CSV
   ============================================================ */
const BRAND_MODELS = {
  "Audi":          ["A4","Q5"],
  "BMW":           ["3 Series","5 Series"],
  "Ford":          ["EcoSport","Figo Aspire"],
  "Honda":         ["Amaze","City","Civic","Elevate","Jazz","WR-V"],
  "Hyundai":       ["Alcazar","Aura","Creta","Exter","Grand i10","Grand i10 Nios","Tucson","Venue","Verna","i20"],
  "Jeep":          ["Compass"],
  "Kia":           ["Carens","Seltos","Sonet"],
  "MG":            ["Astor","Hector","ZS EV"],
  "Mahindra":      ["Bolero","Scorpio","Scorpio-N","Thar","XUV300","XUV400 EV","XUV700"],
  "Maruti":        ["Alto K10","Baleno","Celerio","Ciaz","Dzire","Ertiga","Fronx","Grand Vitara","Jimny","S-Presso","Swift","Vitara Brezza","Wagon R","XL6"],
  "Mercedes-Benz": ["C-Class","E-Class"],
  "Nissan":        ["Kicks","Magnite"],
  "Renault":       ["Duster","Kiger","Kwid","Triber"],
  "Skoda":         ["Kodiaq","Kushaq","Rapid","Slavia"],
  "Tata":          ["Altroz","Harrier","Nexon","Nexon EV","Punch","Safari","Tiago","Tigor"],
  "Toyota":        ["Camry","Corolla Altis","Fortuner","Glanza","Hyryder","Innova Crysta","Urban Cruiser"],
  "Volkswagen":    ["Ameo","Polo","Taigun","Virtus"]
};

/* ============================================================
   2. MODEL — Ridge Regression coefficients (ported from Python)
   
   Preprocessing pipeline (matches notebook exactly):
   - Drop: variant, listing_date, city, state, source_platform, model_name
   - Engineer: vehicle_age = listing_year - registration_year
   - One-hot encode with drop_first=True
   - StandardScaler normalize → Ridge regression predict
   ============================================================ */
const MODEL = {
  intercept: 654500.40,

  // Feature means (from StandardScaler fit on training data)
  means: {
    manufacturing_year: 2020.50,
    km_driven:          56789.85,
    mileage:            29.06,
    engine_cc:          1381.24,
    registration_year:  2020.73,
    listing_year:       2024.24,
    vehicle_age:        3.51,
    "car_brand_BMW":           0.00586,
    "car_brand_Ford":          0.03012,
    "car_brand_Honda":         0.07942,
    "car_brand_Hyundai":       0.16018,
    "car_brand_Jeep":          0.01759,
    "car_brand_Kia":           0.03945,
    "car_brand_MG":            0.02692,
    "car_brand_Mahindra":      0.04771,
    "car_brand_Maruti":        0.21855,
    "car_brand_Mercedes-Benz": 0.00613,
    "car_brand_Nissan":        0.02052,
    "car_brand_Renault":       0.05144,
    "car_brand_Skoda":         0.03412,
    "car_brand_Tata":          0.13380,
    "car_brand_Toyota":        0.08102,
    "car_brand_Volkswagen":    0.04051,
    "fuel_type_Diesel":        0.15458,
    "fuel_type_Electric":      0.02612,
    "fuel_type_Hybrid":        0.00400,
    "fuel_type_Petrol":        0.75213,
    "transmission_Manual":     0.76946,
    "owner_type_2nd":          0.26946,
    "owner_type_3rd+":         0.06930,
    "seller_type_Individual":  0.27159,
    "insurance_validity_Comprehensive":  0.34568,
    "insurance_validity_Expired":        0.10208,
    "insurance_validity_Third Party":    0.25320,
    "insurance_validity_Valid till 2025":0.10448,
    "insurance_validity_Valid till 2026":0.11754,
    "insurance_validity_Valid till 2027":0.04717
  },

  // Feature standard deviations (from StandardScaler)
  stds: {
    manufacturing_year: 2.7506,
    km_driven:          36248.46,
    mileage:            59.989,
    engine_cc:          395.937,
    registration_year:  2.7520,
    listing_year:       0.9434,
    vehicle_age:        2.7703,
    "car_brand_BMW":           0.07635,
    "car_brand_Ford":          0.17091,
    "car_brand_Honda":         0.27040,
    "car_brand_Hyundai":       0.36677,
    "car_brand_Jeep":          0.13146,
    "car_brand_Kia":           0.19465,
    "car_brand_MG":            0.16185,
    "car_brand_Mahindra":      0.21315,
    "car_brand_Maruti":        0.41326,
    "car_brand_Mercedes-Benz": 0.07805,
    "car_brand_Nissan":        0.14178,
    "car_brand_Renault":       0.22089,
    "car_brand_Skoda":         0.18152,
    "car_brand_Tata":          0.34043,
    "car_brand_Toyota":        0.27287,
    "car_brand_Volkswagen":    0.19716,
    "fuel_type_Diesel":        0.36151,
    "fuel_type_Electric":      0.15949,
    "fuel_type_Hybrid":        0.06310,
    "fuel_type_Petrol":        0.43177,
    "transmission_Manual":     0.42118,
    "owner_type_2nd":          0.44368,
    "owner_type_3rd+":         0.25396,
    "seller_type_Individual":  0.44478,
    "insurance_validity_Comprehensive":  0.47559,
    "insurance_validity_Expired":        0.30275,
    "insurance_validity_Third Party":    0.43484,
    "insurance_validity_Valid till 2025":0.30588,
    "insurance_validity_Valid till 2026":0.32206,
    "insurance_validity_Valid till 2027":0.21201
  },

  // Ridge regression coefficients
  coefs: {
    manufacturing_year:  96361.76,
    km_driven:          -16962.94,
    mileage:             29589.03,
    engine_cc:          219320.22,
    registration_year:   37338.99,
    listing_year:        16630.65,
    vehicle_age:        -31428.57,
    "car_brand_BMW":           25995.79,
    "car_brand_Ford":         -65074.50,
    "car_brand_Honda":        -59116.08,
    "car_brand_Hyundai":      -94769.00,
    "car_brand_Jeep":         -15599.71,
    "car_brand_Kia":          -35100.71,
    "car_brand_MG":           -28304.60,
    "car_brand_Mahindra":     -83729.89,
    "car_brand_Maruti":      -112537.20,
    "car_brand_Mercedes-Benz": 98442.30,
    "car_brand_Nissan":       -44197.48,
    "car_brand_Renault":      -67988.75,
    "car_brand_Skoda":         -2574.56,
    "car_brand_Tata":        -108695.61,
    "car_brand_Toyota":       -93815.19,
    "car_brand_Volkswagen":   -29687.82,
    "fuel_type_Diesel":        24878.91,
    "fuel_type_Electric":      30078.42,
    "fuel_type_Hybrid":        33650.81,
    "fuel_type_Petrol":         5035.57,
    "transmission_Manual":   -104283.75,
    "owner_type_2nd":             702.48,
    "owner_type_3rd+":            422.37,
    "seller_type_Individual":   -1633.00,
    "insurance_validity_Comprehensive":   -1320.36,
    "insurance_validity_Expired":          4041.09,
    "insurance_validity_Third Party":     -1410.71,
    "insurance_validity_Valid till 2025":  1840.25,
    "insurance_validity_Valid till 2026":   763.70,
    "insurance_validity_Valid till 2027":  4266.44
  }
};

/**
 * Predict price using Ridge Regression
 * - Builds one-hot encoded feature vector (matching notebook's get_dummies with drop_first=True)
 * - Scales using StandardScaler parameters
 * - Returns predicted price in INR
 */
function predictPrice(input) {
  const {
    car_brand, manufacturing_year, km_driven, fuel_type,
    transmission, owner_type, seller_type, mileage,
    engine_cc, insurance_validity, registration_year, listing_year
  } = input;

  // Derived feature
  const vehicle_age = listing_year - registration_year;

  // Build raw feature vector
  const raw = {
    manufacturing_year: +manufacturing_year,
    km_driven:          +km_driven,
    mileage:            +mileage,
    engine_cc:          +engine_cc,
    registration_year:  +registration_year,
    listing_year:       +listing_year,
    vehicle_age:        vehicle_age,

    // Brand one-hot (Audi is the dropped reference category)
    "car_brand_BMW":           car_brand === "BMW"           ? 1 : 0,
    "car_brand_Ford":          car_brand === "Ford"          ? 1 : 0,
    "car_brand_Honda":         car_brand === "Honda"         ? 1 : 0,
    "car_brand_Hyundai":       car_brand === "Hyundai"       ? 1 : 0,
    "car_brand_Jeep":          car_brand === "Jeep"          ? 1 : 0,
    "car_brand_Kia":           car_brand === "Kia"           ? 1 : 0,
    "car_brand_MG":            car_brand === "MG"            ? 1 : 0,
    "car_brand_Mahindra":      car_brand === "Mahindra"      ? 1 : 0,
    "car_brand_Maruti":        car_brand === "Maruti"        ? 1 : 0,
    "car_brand_Mercedes-Benz": car_brand === "Mercedes-Benz" ? 1 : 0,
    "car_brand_Nissan":        car_brand === "Nissan"        ? 1 : 0,
    "car_brand_Renault":       car_brand === "Renault"       ? 1 : 0,
    "car_brand_Skoda":         car_brand === "Skoda"         ? 1 : 0,
    "car_brand_Tata":          car_brand === "Tata"          ? 1 : 0,
    "car_brand_Toyota":        car_brand === "Toyota"        ? 1 : 0,
    "car_brand_Volkswagen":    car_brand === "Volkswagen"    ? 1 : 0,

    // Fuel type one-hot (CNG is dropped reference)
    "fuel_type_Diesel":   fuel_type === "Diesel"   ? 1 : 0,
    "fuel_type_Electric": fuel_type === "Electric" ? 1 : 0,
    "fuel_type_Hybrid":   fuel_type === "Hybrid"   ? 1 : 0,
    "fuel_type_Petrol":   fuel_type === "Petrol"   ? 1 : 0,

    // Transmission (Automatic is dropped)
    "transmission_Manual": transmission === "Manual" ? 1 : 0,

    // Owner type (1st is dropped reference)
    "owner_type_2nd":  owner_type === "2nd"  ? 1 : 0,
    "owner_type_3rd+": owner_type === "3rd+" ? 1 : 0,

    // Seller type (Dealer is dropped)
    "seller_type_Individual": seller_type === "Individual" ? 1 : 0,

    // Insurance (3rd Party Only is dropped reference)
    "insurance_validity_Comprehensive":   insurance_validity === "Comprehensive"   ? 1 : 0,
    "insurance_validity_Expired":         insurance_validity === "Expired"         ? 1 : 0,
    "insurance_validity_Third Party":     insurance_validity === "Third Party"     ? 1 : 0,
    "insurance_validity_Valid till 2025": insurance_validity === "Valid till 2025" ? 1 : 0,
    "insurance_validity_Valid till 2026": insurance_validity === "Valid till 2026" ? 1 : 0,
    "insurance_validity_Valid till 2027": insurance_validity === "Valid till 2027" ? 1 : 0,
  };

  // Compute scaled dot product
  let prediction = MODEL.intercept;
  for (const [feature, coef] of Object.entries(MODEL.coefs)) {
    const val   = raw[feature] !== undefined ? raw[feature] : 0;
    const mean  = MODEL.means[feature] || 0;
    const std   = MODEL.stds[feature]  || 1;
    const scaled = (val - mean) / std;
    prediction += coef * scaled;
  }

  return Math.max(50000, Math.round(prediction));
}

/* ============================================================
   3. BACKGROUND CANVAS (Particle grid animation)
   ============================================================ */
function initBackground() {
  const canvas = document.getElementById('bg-canvas');
  const ctx = canvas.getContext('2d');
  let W, H, particles;

  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }

  class Particle {
    constructor() { this.reset(); }
    reset() {
      this.x  = Math.random() * W;
      this.y  = Math.random() * H;
      this.vx = (Math.random() - 0.5) * 0.3;
      this.vy = (Math.random() - 0.5) * 0.3;
      this.r  = Math.random() * 1.5 + 0.3;
      this.a  = Math.random() * 0.5 + 0.1;
    }
    update() {
      this.x += this.vx;
      this.y += this.vy;
      if (this.x < 0 || this.x > W || this.y < 0 || this.y > H) this.reset();
    }
    draw() {
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(0, 212, 255, ${this.a})`;
      ctx.fill();
    }
  }

  function init() {
    resize();
    particles = Array.from({ length: 120 }, () => new Particle());
  }

  function drawConnections() {
    const maxDist = 120;
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < maxDist) {
          const alpha = (1 - dist / maxDist) * 0.12;
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.strokeStyle = `rgba(0, 212, 255, ${alpha})`;
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }
    }
  }

  function loop() {
    ctx.clearRect(0, 0, W, H);
    particles.forEach(p => { p.update(); p.draw(); });
    drawConnections();
    requestAnimationFrame(loop);
  }

  window.addEventListener('resize', resize);
  init();
  loop();
}

/* ============================================================
   4. 3D CAR SCENE (Three.js via CDN)
   ============================================================ */
function initCarScene() {
  const canvas = document.getElementById('car-canvas');
  if (!canvas) return;

  // Try to load Three.js from CDN, fall back to CSS car
  const script = document.createElement('script');
  script.src = 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js';
  script.onload  = () => buildThreeCar(canvas);
  script.onerror = () => buildCSSCar(canvas);
  document.head.appendChild(script);
}

function buildThreeCar(canvas) {
  const THREE = window.THREE;
  const W = canvas.offsetWidth || 700;
  const H = canvas.offsetHeight || 420;

  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
  renderer.setSize(W, H);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  const scene  = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, W / H, 0.1, 100);
  camera.position.set(4, 2, 6);
  camera.lookAt(0, 0, 0);

  // Lighting
  const ambient = new THREE.AmbientLight(0x001a33, 2);
  scene.add(ambient);

  const pointCyan = new THREE.PointLight(0x00d4ff, 3, 20);
  pointCyan.position.set(-4, 4, 4);
  scene.add(pointCyan);

  const pointPurple = new THREE.PointLight(0x7c3aed, 2, 20);
  pointPurple.position.set(4, 2, -4);
  scene.add(pointPurple);

  const rimLight = new THREE.DirectionalLight(0x0ea5e9, 1.5);
  rimLight.position.set(0, 8, -6);
  scene.add(rimLight);

  // Car body material
  const carMat = new THREE.MeshStandardMaterial({
    color: 0x0a1628,
    metalness: 0.9,
    roughness: 0.2,
    envMapIntensity: 1.0
  });

  const glassMat = new THREE.MeshStandardMaterial({
    color: 0x00d4ff,
    metalness: 0.2,
    roughness: 0.05,
    opacity: 0.35,
    transparent: true
  });

  const accentMat = new THREE.MeshStandardMaterial({
    color: 0x00d4ff,
    emissive: 0x00d4ff,
    emissiveIntensity: 0.8,
    metalness: 0.1,
    roughness: 0.1
  });

  const wheelMat = new THREE.MeshStandardMaterial({
    color: 0x111827,
    metalness: 0.7,
    roughness: 0.3
  });

  // Build car geometry group
  const car = new THREE.Group();

  // Main body
  const bodyGeo = new THREE.BoxGeometry(3.8, 0.7, 1.7);
  const body = new THREE.Mesh(bodyGeo, carMat);
  body.position.y = 0.3;
  car.add(body);

  // Roof/cabin
  const cabinGeo = new THREE.BoxGeometry(2.2, 0.55, 1.5);
  const cabin = new THREE.Mesh(cabinGeo, carMat);
  cabin.position.set(0.1, 0.9, 0);
  car.add(cabin);

  // Windshield
  const windGeo = new THREE.BoxGeometry(0.05, 0.5, 1.4);
  const windF = new THREE.Mesh(windGeo, glassMat);
  windF.position.set(1.12, 0.85, 0);
  windF.rotation.z = -0.18;
  car.add(windF);

  const windR = new THREE.Mesh(windGeo, glassMat);
  windR.position.set(-1.12, 0.85, 0);
  windR.rotation.z = 0.18;
  car.add(windR);

  // Side windows
  const sideWinGeo = new THREE.BoxGeometry(1.8, 0.38, 0.04);
  const sideWinL = new THREE.Mesh(sideWinGeo, glassMat);
  sideWinL.position.set(0.1, 0.9, 0.77);
  car.add(sideWinL);
  const sideWinR = sideWinL.clone();
  sideWinR.position.z = -0.77;
  car.add(sideWinR);

  // Neon underline strip
  const neonGeo = new THREE.BoxGeometry(3.6, 0.04, 0.04);
  const neonL = new THREE.Mesh(neonGeo, accentMat);
  neonL.position.set(0, -0.08, 0.87);
  car.add(neonL);
  const neonR = neonL.clone();
  neonR.position.z = -0.87;
  car.add(neonR);

  // Front grill neon
  const grillGeo = new THREE.BoxGeometry(0.04, 0.15, 1.5);
  const grill = new THREE.Mesh(grillGeo, accentMat);
  grill.position.set(1.92, 0.18, 0);
  car.add(grill);

  // Headlights
  const headlightGeo = new THREE.BoxGeometry(0.05, 0.1, 0.3);
  const hlMat = new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0xaaddff, emissiveIntensity: 2 });
  const hlL = new THREE.Mesh(headlightGeo, hlMat);
  hlL.position.set(1.93, 0.28, 0.55);
  car.add(hlL);
  const hlR = hlL.clone();
  hlR.position.z = -0.55;
  car.add(hlR);

  // Wheels
  function makeWheel(x, z) {
    const wg = new THREE.Group();
    const tire = new THREE.Mesh(
      new THREE.CylinderGeometry(0.38, 0.38, 0.25, 24),
      wheelMat
    );
    tire.rotation.x = Math.PI / 2;
    const rim = new THREE.Mesh(
      new THREE.CylinderGeometry(0.22, 0.22, 0.27, 6),
      new THREE.MeshStandardMaterial({ color: 0x334155, metalness: 0.9, roughness: 0.1 })
    );
    rim.rotation.x = Math.PI / 2;
    const hubcap = new THREE.Mesh(
      new THREE.CylinderGeometry(0.07, 0.07, 0.28, 6),
      accentMat
    );
    hubcap.rotation.x = Math.PI / 2;
    wg.add(tire); wg.add(rim); wg.add(hubcap);
    wg.position.set(x, -0.12, z);
    return wg;
  }

  car.add(makeWheel( 1.3,  0.9));
  car.add(makeWheel( 1.3, -0.9));
  car.add(makeWheel(-1.3,  0.9));
  car.add(makeWheel(-1.3, -0.9));

  // Ground reflection plane
  const groundGeo = new THREE.PlaneGeometry(12, 8);
  const groundMat = new THREE.MeshStandardMaterial({
    color: 0x000a1a,
    metalness: 0.9,
    roughness: 0.2,
    opacity: 0.6,
    transparent: true
  });
  const ground = new THREE.Mesh(groundGeo, groundMat);
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = -0.5;
  scene.add(ground);

  // Grid lines on ground
  const gridHelper = new THREE.GridHelper(12, 20, 0x001a33, 0x001a33);
  gridHelper.position.y = -0.49;
  scene.add(gridHelper);

  car.position.y = 0.0;
  car.rotation.y = 0.4;
  scene.add(car);

  // Animation
  let t = 0;
  function animate() {
    requestAnimationFrame(animate);
    t += 0.008;
    car.rotation.y = 0.4 + Math.sin(t * 0.5) * 0.15;
    car.position.y = Math.sin(t) * 0.04;

    // Pulse neon
    const pulse = 0.5 + 0.5 * Math.sin(t * 2);
    accentMat.emissiveIntensity = 0.5 + pulse * 0.5;
    pointCyan.intensity = 2.5 + pulse;

    renderer.render(scene, camera);
  }
  animate();

  // Resize
  window.addEventListener('resize', () => {
    const W2 = canvas.offsetWidth;
    const H2 = canvas.offsetHeight;
    renderer.setSize(W2, H2);
    camera.aspect = W2 / H2;
    camera.updateProjectionMatrix();
  });
}

function buildCSSCar(canvas) {
  // Fallback: draw a stylized 2D car with canvas API
  const ctx = canvas.getContext('2d');
  let t = 0;

  function draw() {
    const W = canvas.width  = canvas.offsetWidth;
    const H = canvas.height = canvas.offsetHeight;
    ctx.clearRect(0, 0, W, H);

    const cx = W / 2, cy = H / 2;
    t += 0.02;
    const bob = Math.sin(t) * 5;

    // Glow shadow
    ctx.shadowBlur  = 40;
    ctx.shadowColor = '#00d4ff';

    // Car body
    ctx.fillStyle = '#0d1b2e';
    roundRect(ctx, cx - 180, cy + bob, 360, 90, 20);
    ctx.fill();

    // Cabin
    ctx.fillStyle = '#0a1520';
    roundRect(ctx, cx - 110, cy - 50 + bob, 220, 80, 14);
    ctx.fill();

    // Windows
    ctx.fillStyle = 'rgba(0,212,255,0.25)';
    ctx.shadowBlur = 10;
    roundRect(ctx, cx - 90, cy - 44 + bob, 80, 58, 8);
    ctx.fill();
    roundRect(ctx, cx + 10, cy - 44 + bob, 80, 58, 8);
    ctx.fill();

    // Neon strip
    ctx.shadowBlur  = 20;
    ctx.shadowColor = '#00d4ff';
    ctx.strokeStyle = '#00d4ff';
    ctx.lineWidth   = 3;
    ctx.beginPath();
    ctx.moveTo(cx - 175, cy + 78 + bob);
    ctx.lineTo(cx + 175, cy + 78 + bob);
    ctx.stroke();

    // Headlights
    ctx.fillStyle = 'rgba(200,240,255,0.9)';
    ctx.shadowBlur = 30;
    ctx.shadowColor = '#aaddff';
    ctx.beginPath(); ctx.ellipse(cx + 172, cy + 40 + bob, 10, 6, 0, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); ctx.ellipse(cx - 172, cy + 40 + bob, 10, 6, 0, 0, Math.PI * 2); ctx.fill();

    // Wheels
    ctx.shadowBlur = 15;
    ctx.shadowColor = '#7c3aed';
    drawWheel(ctx, cx - 110, cy + 112 + bob);
    drawWheel(ctx, cx + 110, cy + 112 + bob);

    // Ground reflection
    ctx.shadowBlur = 0;
    const grad = ctx.createLinearGradient(0, cy + 135, 0, cy + 165);
    grad.addColorStop(0, 'rgba(0,212,255,0.12)');
    grad.addColorStop(1, 'rgba(0,212,255,0)');
    ctx.fillStyle = grad;
    ctx.fillRect(cx - 200, cy + 135 + bob * 0.3, 400, 30);

    requestAnimationFrame(draw);
  }

  function drawWheel(ctx, x, y) {
    ctx.beginPath(); ctx.arc(x, y, 32, 0, Math.PI * 2);
    ctx.fillStyle = '#111827'; ctx.fill();
    ctx.strokeStyle = '#334155'; ctx.lineWidth = 4; ctx.stroke();
    ctx.beginPath(); ctx.arc(x, y, 14, 0, Math.PI * 2);
    ctx.strokeStyle = '#00d4ff'; ctx.lineWidth = 2; ctx.stroke();
    ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fillStyle = '#00d4ff'; ctx.fill();
  }

  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.arcTo(x + w, y, x + w, y + r, r);
    ctx.lineTo(x + w, y + h - r);
    ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
    ctx.lineTo(x + r, y + h);
    ctx.arcTo(x, y + h, x, y + h - r, r);
    ctx.lineTo(x, y + r);
    ctx.arcTo(x, y, x + r, y, r);
    ctx.closePath();
  }

  draw();
}

/* ============================================================
   5. FORM POPULATION
   ============================================================ */
function populateForm() {
  // Brands
  const brandSel = document.getElementById('car_brand');
  Object.keys(BRAND_MODELS).sort().forEach(brand => {
    const opt = document.createElement('option');
    opt.value = brand; opt.textContent = brand;
    brandSel.appendChild(opt);
  });

  // Years (manufacturing & registration: 2012–2025, listing: 2023–2026)
  const mfgSel  = document.getElementById('manufacturing_year');
  const regSel  = document.getElementById('registration_year');
  const lstSel  = document.getElementById('listing_year');

  for (let y = 2025; y >= 2012; y--) {
    [mfgSel, regSel].forEach(sel => {
      const o = document.createElement('option');
      o.value = y; o.textContent = y;
      sel.appendChild(o);
    });
  }

  [2026, 2025, 2024, 2023].forEach(y => {
    const o = document.createElement('option');
    o.value = y; o.textContent = y;
    lstSel.appendChild(o);
  });

  // Dependent brand→model dropdown
  brandSel.addEventListener('change', () => {
    const modelSel = document.getElementById('model_name');
    modelSel.innerHTML = '<option value="">Select Model</option>';
    const brand = brandSel.value;
    if (brand && BRAND_MODELS[brand]) {
      BRAND_MODELS[brand].forEach(m => {
        const opt = document.createElement('option');
        opt.value = m; opt.textContent = m;
        modelSel.appendChild(opt);
      });
      modelSel.disabled = false;
    } else {
      modelSel.disabled = true;
    }
    clearError('car_brand');
  });

  // Vehicle age calculation
  function updateVehicleAge() {
    const reg = +regSel.value;
    const lst = +lstSel.value;
    if (reg && lst) {
      const age = lst - reg;
      const display = document.getElementById('vehicle-age-display');
      const val     = document.getElementById('vehicle-age-value');
      display.style.display = 'flex';
      val.textContent = age >= 0 ? age : '⚠ Invalid';
      val.style.color = age < 0 ? 'var(--danger)' : 'var(--neon-cyan)';
    }
  }
  regSel.addEventListener('change', updateVehicleAge);
  lstSel.addEventListener('change', updateVehicleAge);
}

/* ============================================================
   6. FORM VALIDATION
   ============================================================ */
function validateForm() {
  let valid = true;

  const fields = [
    { id: 'car_brand',          msg: 'Please select a brand.' },
    { id: 'model_name',         msg: 'Please select a model.' },
    { id: 'manufacturing_year', msg: 'Please select manufacturing year.' },
    { id: 'registration_year',  msg: 'Please select registration year.' },
    { id: 'listing_year',       msg: 'Please select listing year.' },
    { id: 'fuel_type',          msg: 'Please select fuel type.' },
    { id: 'transmission',       msg: 'Please select transmission.' },
    { id: 'owner_type',         msg: 'Please select owner type.' },
    { id: 'seller_type',        msg: 'Please select seller type.' },
  ];

  fields.forEach(({ id, msg }) => {
    const el = document.getElementById(id);
    if (!el || !el.value) {
      showError(id, msg); valid = false;
    } else {
      clearError(id);
    }
  });

  // Numeric fields
  const numFields = [
    { id: 'engine_cc', min: 600,  max: 3500,  msg: 'Enter engine CC between 600–3500.' },
    { id: 'mileage',   min: 5,    max: 500,   msg: 'Enter mileage between 5–500.' },
    { id: 'km_driven', min: 0,    max: 300000,msg: 'Enter km driven between 0–3,00,000.' },
  ];

  numFields.forEach(({ id, min, max, msg }) => {
    const el = document.getElementById(id);
    const v  = parseFloat(el.value);
    if (!el.value || isNaN(v) || v < min || v > max) {
      showError(id, msg);
      el.classList.add('error');
      valid = false;
    } else {
      clearError(id);
      el.classList.remove('error');
    }
  });

  // Insurance radio
  const insuranceSel = document.querySelector('input[name="insurance_validity"]:checked');
  if (!insuranceSel) {
    showError('insurance_validity', 'Please select insurance validity.');
    valid = false;
  } else {
    clearError('insurance_validity');
  }

  // Year logic
  const mfg = +document.getElementById('manufacturing_year').value;
  const reg = +document.getElementById('registration_year').value;
  const lst = +document.getElementById('listing_year').value;

  if (mfg && reg && mfg > reg) {
    showError('registration_year', 'Registration year cannot be before manufacturing year.');
    valid = false;
  }
  if (reg && lst && lst < reg) {
    showError('listing_year', 'Listing year cannot be before registration year.');
    valid = false;
  }

  return valid;
}

function showError(id, msg) {
  const errEl = document.getElementById(`err-${id}`);
  if (errEl) errEl.textContent = msg;
  const el = document.getElementById(id);
  if (el) el.classList.add('error');
}

function clearError(id) {
  const errEl = document.getElementById(`err-${id}`);
  if (errEl) errEl.textContent = '';
  const el = document.getElementById(id);
  if (el) el.classList.remove('error');
}

function clearAllErrors() {
  document.querySelectorAll('.err-msg').forEach(el => el.textContent = '');
  document.querySelectorAll('.error').forEach(el => el.classList.remove('error'));
}

/* ============================================================
   7. RESULT RENDERING
   ============================================================ */
function showResult(price, input) {
  const carName = `${input.car_brand} ${input.model_name}`;
  const vehicle_age = (+input.listing_year) - (+input.registration_year);
  const low  = Math.round(price * 0.85);
  const high = Math.round(price * 1.15);

  // Show the result panel
  document.getElementById('result-idle').style.display   = 'none';
  const output = document.getElementById('result-output');
  output.style.display = 'flex';

  // Car name
  document.getElementById('result-car-name').textContent =
    `${carName} (${input.manufacturing_year})`;

  // Price animation
  animateNumber('result-price', price);

  // Range
  document.getElementById('range-low').textContent  = '₹' + formatINR(low);
  document.getElementById('range-high').textContent = '₹' + formatINR(high);

  // Market position: 0–100% relative to dataset range (83500–4743000)
  const pct = Math.min(100, Math.max(5,
    ((price - 83500) / (4743000 - 83500)) * 100
  ));
  document.getElementById('range-fill').style.width  = `${pct}%`;
  document.getElementById('range-thumb').style.left  = `${pct}%`;

  // Metrics
  const confidence = price > 100000 ? (price > 3000000 ? 'High' : 'Very High') : 'Moderate';
  const marketPos  = price < 400000 ? 'Budget' :
                     price < 800000 ? 'Mid-Range' :
                     price < 1500000 ? 'Premium' : 'Luxury';

  document.getElementById('metric-confidence').textContent = confidence;
  document.getElementById('metric-market').textContent     = marketPos;
  document.getElementById('metric-age').textContent        = `${vehicle_age} yr${vehicle_age !== 1 ? 's' : ''}`;

  // Insights
  const insights = generateInsights(input, price, vehicle_age);
  const insightEl = document.getElementById('result-insights');
  insightEl.innerHTML = insights.map(({ icon, text }) =>
    `<div class="insight-item"><span class="insight-icon">${icon}</span><span>${text}</span></div>`
  ).join('');

  // Scroll result into view on mobile
  if (window.innerWidth < 1100) {
    output.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
}

function generateInsights(input, price, age) {
  const insights = [];
  const km = +input.km_driven;

  if (input.transmission === 'Automatic') {
    insights.push({ icon: '⚙️', text: 'Automatic transmission adds a significant premium over manual variants.' });
  }
  if (['Electric', 'Hybrid'].includes(input.fuel_type)) {
    insights.push({ icon: '🔋', text: `${input.fuel_type} vehicles command higher resale value in current market.` });
  }
  if (km > 80000) {
    insights.push({ icon: '📉', text: 'High mileage (>80,000 km) reduces resale value. Consider negotiating.' });
  } else if (km < 20000) {
    insights.push({ icon: '📈', text: 'Low mileage is a strong selling point — expect better resale value.' });
  }
  if (age > 7) {
    insights.push({ icon: '🕐', text: `Older vehicle (${age} years) — depreciation significantly impacts price.` });
  }
  if (['Mercedes-Benz', 'BMW', 'Audi'].includes(input.car_brand)) {
    insights.push({ icon: '💎', text: 'Luxury brand commands premium resale, but maintenance costs are higher.' });
  }
  if (input.insurance_validity === 'Expired') {
    insights.push({ icon: '⚠️', text: 'Expired insurance reduces buyer confidence. Renewing before sale is advised.' });
  }
  if (input.owner_type === '1st') {
    insights.push({ icon: '✅', text: 'First-owner vehicles are highly preferred by buyers in the used car market.' });
  }

  return insights.slice(0, 4);
}

function animateNumber(id, target) {
  const el = document.getElementById(id);
  const start = 0;
  const duration = 1200;
  const startTime = performance.now();

  function update(now) {
    const elapsed = now - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const ease = 1 - Math.pow(1 - progress, 3);
    const current = Math.round(start + (target - start) * ease);
    el.textContent = formatINR(current);
    if (progress < 1) requestAnimationFrame(update);
  }
  requestAnimationFrame(update);
}

function formatINR(n) {
  if (n >= 10000000) return (n / 10000000).toFixed(2) + ' Cr';
  if (n >= 100000)   return (n / 100000).toFixed(2) + ' L';
  return n.toLocaleString('en-IN');
}

/* ============================================================
   8. FORM SUBMISSION & RESET
   ============================================================ */
function initForm() {
  const form       = document.getElementById('prediction-form');
  const predictBtn = document.getElementById('predict-btn');
  const resetBtn   = document.getElementById('reset-btn');

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!validateForm()) return;

    // Loading state
    predictBtn.querySelector('.btn-text').style.display  = 'none';
    predictBtn.querySelector('.btn-loader').style.display = 'flex';
    predictBtn.querySelector('.btn-arrow').style.display  = 'none';
    predictBtn.disabled = true;

    // Simulate brief processing delay for UX
    await new Promise(r => setTimeout(r, 900));

    const input = {
      car_brand:          document.getElementById('car_brand').value,
      model_name:         document.getElementById('model_name').value,
      manufacturing_year: document.getElementById('manufacturing_year').value,
      registration_year:  document.getElementById('registration_year').value,
      listing_year:       document.getElementById('listing_year').value,
      fuel_type:          document.getElementById('fuel_type').value,
      transmission:       document.getElementById('transmission').value,
      owner_type:         document.getElementById('owner_type').value,
      seller_type:        document.getElementById('seller_type').value,
      engine_cc:          document.getElementById('engine_cc').value,
      mileage:            document.getElementById('mileage').value,
      km_driven:          document.getElementById('km_driven').value,
      insurance_validity: document.querySelector('input[name="insurance_validity"]:checked').value,
    };

    const price = predictPrice(input);

    // Restore button
    predictBtn.querySelector('.btn-text').style.display  = 'inline';
    predictBtn.querySelector('.btn-loader').style.display = 'none';
    predictBtn.querySelector('.btn-arrow').style.display  = 'inline';
    predictBtn.disabled = false;

    showResult(price, input);
  });

  resetBtn.addEventListener('click', () => {
    form.reset();
    document.getElementById('model_name').disabled = true;
    document.getElementById('model_name').innerHTML = '<option value="">Select Model</option>';
    document.getElementById('vehicle-age-display').style.display = 'none';
    clearAllErrors();
    document.getElementById('result-idle').style.display  = 'flex';
    document.getElementById('result-output').style.display = 'none';
  });

  // Clear errors on input
  form.querySelectorAll('input, select').forEach(el => {
    el.addEventListener('input', () => clearError(el.id || el.name));
    el.addEventListener('change', () => clearError(el.id || el.name));
  });
}

/* ============================================================
   9. INTERSECTION OBSERVER — scroll animations
   ============================================================ */
function initScrollAnimations() {
  const targets = document.querySelectorAll(
    '.step-card, .feature-card, .form-section, .section-header'
  );

  targets.forEach(el => el.classList.add('fade-in'));

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.1 });

  targets.forEach(el => observer.observe(el));
}

/* ============================================================
   10. INIT
   ============================================================ */
document.addEventListener('DOMContentLoaded', () => {
  initBackground();
  initCarScene();
  populateForm();
  initForm();
  initScrollAnimations();
});
