<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>RECYKUY</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet"/>
  <style>
    body {
      background: linear-gradient(to bottom right, #0c0c0c, #1e1e1e);
      color: white;
      font-family: 'Segoe UI', sans-serif;
    }
    nav {
     padding: 1rem 2rem;
     position: sticky;
     top: 0;
     z-index: 1000;
     background-color: #2e7d32; /* warna solid biar kontras saat scroll */
     transition: all 0.3s ease;
     transition: background-color 0.3s ease;
    }
    .navbar.scrolled {
  background-color: #2e7d32 !important;
  box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
}
    .nav-link {
      color: white !important;
      margin-right: 1rem;
    }
    .btn-quote {
      background-color: #34A853;
      color: white;
      border-radius: 20px;
      padding: 0.5rem 1rem;
      font-weight: 500;
    }
    .hero {
        padding: 5rem 2rem 2rem 2rem;
  background-image: url('{{ asset('css/Banner 1.jpg') }}');
  background-size: cover;
  background-position: center;
  position: relative;
  z-index: 1;
  color: white;
    }
    .hero::before {
        content: "";
  position: absolute;
  top: 0; left: 0; right: 0; bottom: 0;
  background: rgba(0,0,0,0.6); /* dark overlay */
  z-index: -1;
    }
    .hero h1 {
      font-size: 3rem;
    }
    .hero h1 span {
        color: #34A853;
    }
    .features {
      padding: 2rem;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 2rem;
    }
    .feature-box {
      background: rgba(255, 255, 255, 0.05);
      padding: 1rem;
      border-radius: 15px;
      text-align: center;
    }
    .feature-box i {
      font-size: 2rem;
      margin-bottom: 0.5rem;
    }
    .whatsapp-button {
      position: fixed;
      bottom: 20px;
      right: 20px;
      background-color: #25D366;
      color: white;
      border-radius: 50%;
      width: 50px;
      height: 50px;
      font-size: 24px;
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 4px 6px rgba(0,0,0,0.3);
      z-index: 999;
    }

    body {
      font-family: 'Segoe UI', sans-serif;
      background-color: #f9f9f5;
      margin: 0;
      padding: 0;
    }
    .commitment-section {
  background-color: #ffffff;
  color: #222222;
  padding: 5rem 2rem;
}
.commitment-section .value-icon {
  background: linear-gradient(to right, #84d32a, #43a047);
}
.commitment-section h5 {
  color: #2e7d32;
}
    .navbar {
      transition: all 0.3s ease;
    }

    .btn-quote {
      background-color: #4CAF50;
      color: white;
      border-radius: 30px;
      padding: 6px 16px;
    }
    .btn-quote:hover {
      background-color: #45a049;
    }
    .value-icon {
      width: 60px;
      height: 60px;
      background: linear-gradient(to right, #84d32a, #43a047);
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      margin: 0 auto 20px;
    }
    .value-icon i {
      font-size: 28px;
      color: white;
    }
    .running-text {
  overflow: hidden;
  white-space: nowrap;
  position: relative;
  font-weight: 600;
  font-size: 48px;
  padding: 40px 0;
  color: transparent;
  background-color: white;
  -webkit-text-stroke: 1px #4CAF50;
}



.running-text .scrolling-content {
  display: inline-block;
  animation: scroll-left 30s linear infinite;
}
.team-photo {
  width: 100%;
  height: 400px;
  object-fit: cover;
  object-position: center;
  border-top-left-radius: 0.375rem; /* menyamakan sudut bootstrap */
  border-top-right-radius: 0.375rem;
}

.cta-section {
  background-color: #f7f7f7;
  color: #111;
  padding: 4rem 2rem;
}

.cta-title {
  font-size: 2rem;
  margin-bottom: 1.5rem;
}

.cta-section .btn-success {
  background-color: #2e7d32;
  border: none;
  padding: 0.75rem 2rem;
  border-radius: 8px;
  font-weight: 600;
}

.cta-section .btn-success:hover {
  background-color: #27662b;
}
@keyframes scroll-left {
  0% { transform: translateX(0); }
  100% { transform: translateX(-50%); }
}

/* Footer Styles */
/* Footer Styles */
.footer {
  background: linear-gradient(to bottom right, #0c0c0c, #1e1e1e, #2e7d32); /* Black to dark green gradient */
  color: white;
  padding: 4rem 0;
}

.footer-logo {
  max-width: 100%;
  height: auto;
  margin-bottom: 1rem;
}

.footer-tagline {
  font-size: 2rem;
  line-height: 1.2;
  margin-bottom: 0;
}

.footer-tagline span {
  color: #34A853; /* Green accent color */
}

.address {
  margin-bottom: 1rem;
}

.social-links {
  display: flex;
  gap: 1rem;
  margin-top: 1rem;
}

.social-links a {
  color: white;
  text-decoration: none;
  font-size: 1.5rem;
}

.footer-bottom {
  border-top: 1px solid rgba(255, 255, 255, 0.2);
  padding-top: 2rem;
  margin-top: 4rem;
}

.nav-pills .nav-link {
  color: white;
  font-weight: 500;
  margin-right: 1rem;
}

.nav-pills .nav-link:hover {
  color: #34A853; /* Green accent color on hover */
}

.copyright {
  font-size: 0.9rem;
}

  </style>
</head>
<body>

  <!-- Navbar -->
  <nav class="d-flex justify-content-between align-items-center">
    <div class="d-flex align-items-center">
      <img src="{{ asset('css/Logo PolosG.png') }}" alt="Logo" style="height: 50px; margin-right: 10px; object-fit: contain;">
      <strong>RECYKUY</strong>
    </div>
    <div class="d-flex align-items-center">
      <a class="nav-link" href="#">Home</a>
      <a class="nav-link" href="https://wa.me/6289655108829">Contact</a>
      <a class="nav-link" href="https://wa.me/6289655108829"><i class="bi bi-whatsapp"></i> +6289655108829</a>
      <a class="btn btn-quote ms-2" href="#ctaSection">Start Recycling Today</a>
    </div>
  </nav>

  <!--BAGIAN ATAS -->
  <!-- Hero Section -->
  <section class="hero text-white text-center">
    <p class="text-uppercase text-success fw-bold">Pioneering Sustainable Recycling for a Greener Future</p>
    <h1>Limitless Potential<br> Through  <span>Recycling</span></h1>

         <!-- Features Section -->
  <section class="features container text-white" style="padding-top: 100px;">
    <div class="feature-box">
      <i class="bi bi-house-door-fill"></i>
      <p>With trusted experience, we recycle diverse waste materials into valuable, high-quality products.</p>
    </div>
    <div class="feature-box">
      <i class="bi bi-layers-fill"></i>
      <p>We provide information on recycled materials like PET flakes, polypropylene pellets, and polyethylene pellets.</p>
    </div>
    <div class="feature-box">
      <i class="bi bi-cloud-fill"></i>
      <p>Our platform introduces recycled products such as polyester fiber, reusable bags, and large bags.</p>
    </div>
    <div class="feature-box">
      <i class="bi bi-stack"></i>
      <p>We also feature innovative recycled solutions, including geotextiles and other eco-friendly options.</p>
    </div>
    <div class="feature-box">
      <i class="bi bi-building"></i>
      <p>Committed to sustainability, we recommend high-quality and eco-friendly recycling solutions.</p>
    </div>

    <!-- Hero Padding -->
<div style="padding-top: 100px;"></div>

  </section>
  </section>

 

  


<!--BAGIAN KEDUA -->
<!-- Commitment Section -->
<section class="commitment-section text-center">
    <p class="text-success fw-semibold">// OUR COMMITMENTS AND VALUE</p>
    <h2 class="fw-bold">Join us in building <span class="text-success">a brighter tomorrow.</span></h2>
    <a class="btn btn-success mt-3" href="https://wa.me/6289655108829">Contact Us <i class="bi bi-arrow-up-right"></i></a>
  
    <div class="row mt-5">
      <div class="row mt-5">
        <div class="col-md-4 mb-4">
          <div class="value-icon"><i class="bi bi-journal-bookmark-fill"></i></div>
          <h5><strong>01. Education</strong></h5>
          <p>We strive to raise awareness and provide knowledge that helps individuals make informed decisions about recycling and waste management.</p>
        </div>
        <div class="col-md-4 mb-4">
          <div class="value-icon"><i class="bi bi-globe2"></i></div>
          <h5><strong>02. Sustainability First</strong></h5>
          <p>Our platform prioritizes eco-friendly recommendations to promote a greener, more sustainable future for Indonesia.</p>
        </div>
        <div class="col-md-4 mb-4">
          <div class="value-icon"><i class="bi bi-people-fill"></i></div>
          <h5><strong>03. Empowerment</strong></h5>
          <p>We empower communities and individuals to take action through accessible recycling solutions and guidance.</p>
        </div>
    </div>
  </section>

<!-- Running Text -->
<div class="running-text">
    <div class="scrolling-content">
      Education — Sustainability First — Empowerment — Education — Sustainability First — Empowerment —
      Education — Sustainability First — Empowerment — Education — Sustainability First — Empowerment —
    </div>
  </div>

<!-- Meet Our Team Section -->
<section class="text-center py-5" style="background-color: #ffffff; color: #222;">
    <h2 class="fw-bold mb-2">MEET OUR TEAM</h2>
    <p class="text-muted mb-5">Discover the Passionate Team Committed to Solving Indonesia’s Waste Issues</p>
    
    <div class="container">
      <div class="row justify-content-center">
        <!-- Card 1 -->
        <div class="col-md-4 mb-4">
          <div class="card h-100 border-0 shadow-sm">
            <img src="{{ asset('css/Yeba.jpg') }}" class="card-img-top team-photo" alt="Natanael Cristo">
            <div class="card-body text-center">
              <h5 class="card-title fw-bold">Yeba Teo Reinhad Gunawan</h5>
              <p class="text-success">Corporate Strategic of RECYKUY</p>
              <a href="https://www.linkedin.com/in/yebateo" class="btn btn-outline-success" target="_blank" rel="noopener noreferrer">
                <i class="bi bi-linkedin"></i> View LinkedIn Profile
              </a>
            </div>
          </div>
        </div>
  
        <!-- Card 2 -->
        <div class="col-md-4 mb-4">
          <div class="card h-100 border-0 shadow-sm">
            <img src="{{ asset('css/Kenji.jpg') }}" class="card-img-top team-photo" alt="Natanael Cristo">
            <div class="card-body text-center">
              <h5 class="card-title fw-bold">Kenji Tjandra</h5>
              <p class="text-success">Head of RECYKUY</p>
              <a href="https://www.linkedin.com/in/kenji-tjandra-9a7b97330?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app" class="btn btn-outline-success" target="_blank" rel="noopener noreferrer">
                <i class="bi bi-linkedin"></i> View LinkedIn Profile
              </a>
            </div>
          </div>
        </div>
  
       <!-- Card 3 -->
<div class="col-md-4 mb-4">
    <div class="card h-100 border-0 shadow-sm">
        <img src="{{ asset('css/Foto Sayah.jpg') }}" class="card-img-top team-photo" alt="Natanael Cristo">
      <div class="card-body text-center">
        <h5 class="card-title fw-bold">Natanael Cristo</h5>
        <p class="text-success">Chief of  RECYKUY</p>
        <a href="http://www.linkedin.com/in/natanael-cristo" class="btn btn-outline-success" target="_blank" rel="noopener noreferrer">
          <i class="bi bi-linkedin"></i> View LinkedIn Profile
        </a>
      </div>
    </div>
  </div>
      </div>
    </div>
  </section>

  <!-- CTA Section -->
<section id="ctaSection" class="cta-section text-center">
    <h2 class="cta-title fw-bold">Prepared to Take a Smarter Approach to Waste Management?</h2>
  
    <!-- Tombol trigger file input -->
    <button id="actionButton" class="btn btn-success btn-lg mt-3">GET STARTED</button>

    <div id="uploadSection" style="display: none; margin-top: 1rem;">
      <div class="row">
        <!-- Kolom kiri: Gambar -->
        <div class="col-md-6 text-center">
          <input type="file" id="imageInput" accept="image/*" style="display: block; margin: 0 auto;" />
          <div id="imagePreview" style="margin-top: 1.5rem;"></div>
        </div>
    
        <!-- Kolom kanan: Textarea sejajar dengan gambar -->
        <div class="col-md-6 d-flex align-items-center">
          <div style="width: 100%;">
            <label for="responseBox" class="form-label fw-bold">Recycling Suggestion</label>
            <textarea id="responseBox" class="form-control" rows="20" placeholder="System will display response here..."></textarea>
          </div>
        </div>
      </div>
    </div>
    <script>
      document.getElementById("actionButton").addEventListener("click", function () {
        document.getElementById("uploadSection").style.display = "block";
      });
      
      document.getElementById("imageInput").addEventListener("change", function () {
        const file = this.files[0];
        if (file) {
          // Tampilkan gambar preview
          const reader = new FileReader();
          reader.onload = function (e) {
            document.getElementById("imagePreview").innerHTML =
              `<img src="${e.target.result}" alt="Preview" class="img-fluid rounded" style="max-height: 300px;" />`;
          };
          reader.readAsDataURL(file);
      
          // Kirim ke backend Flask lokal
          const formData = new FormData();
          formData.append("file", file);
      
          fetch("http://127.0.0.1:5000/classify", {
            method: "POST",
            body: formData,
          })
            .then(res => res.json())
            .then(data => {
              if (data.output) {
                document.getElementById("responseBox").value = data.output;
              } else {
                document.getElementById("responseBox").value = "Error: " + (data.error || "Unknown error");
              }
            })
            .catch(err => {
              document.getElementById("responseBox").value = "Gagal konek ke backend: " + err.message;
            });
        }
      });
      </script>
  </section>
  
<!-- Footer -->
<footer class="footer">
    <div class="container">
      <div class="row">
        <!-- Logo and Tagline -->
        <!-- Logo and Tagline -->
<div class="col-md-4">
  <img src="{{ asset('css/Logo PolosG.png') }}" alt="Company Logo" class="footer-logo" style="width: 150px; height: auto; margin-top: -35px; margin-left: -40px;" />
  <h2 class="footer-tagline">Leading Recycling Technology<br />for <span class="text-success">Sustainable Future</span></h2>
</div>
  
        <!-- Factory Information -->
        <div class="col-md-4">
          <h3>Our Institute</h3>
          <address>
            <strong>Universitas Bunda Mulia</strong><br />
            Jalan Raya Lodan Raya No.2, Kec.<br />
            Pademangan, RT.12/RW.2, Ancol,<br />
            Jakarta Utara - Indonesia
          </address>
         
        </div>
  
        <!-- Contact Information -->
        <div class="col-md-4">
          <h3>Contact Us</h3>
          <p><strong>Email:</strong> natanaelcristo04@gmail.com</p>
          <p><strong>Phone / Whatsapp:</strong> +6289655108829</p>
        </div>
      </div>
  
      <!-- Bottom Navigation and Copyright -->
      <div class="footer-bottom">
        <div class="row">
          <div class="col-md-6">
            <ul class="nav nav-pills justify-content-start">
              <li class="nav-item"><a class="nav-link" href="#">Home</a></li>
              <li class="nav-item"><a class="nav-link" href="#ctaSection">Services</a></li>
              <li class="nav-item"><a class="nav-link" href="https://wa.me/6289655108829">Contact</a></li>
            </ul>
          </div>
          <div class="col-md-6 text-end">
            <p class="copyright">Copyright © 2025 All Rights Reserved.</p>
          </div>
        </div>
      </div>
    </div>
  </footer>

<!-- JavaScript -->
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
<script>
  // Scroll navbar effect
  window.addEventListener('scroll', function () {
    var navbar = document.getElementById('navbar');
    if (window.scrollY > 50) {
      navbar.classList.add('scrolled');
    } else {
      navbar.classList.remove('scrolled');
    }
  });
</script>

<script>
    const actionButton = document.getElementById("actionButton");
    const imageInput = document.getElementById("imageInput");
    const imagePreview = document.getElementById("imagePreview");
  
    let isUploadMode = false;
  
    actionButton.addEventListener("click", () => {
      if (!isUploadMode) {
        document.getElementById("uploadSection").style.display = "block";
        imageInput.click(); // langsung buka file picker
        actionButton.textContent = "SUBMIT";
        isUploadMode = true;
      } else {
        const file = imageInput.files[0];
        if (file) {
          const reader = new FileReader();
          reader.onload = function (e) {
            imagePreview.innerHTML = `
              <img src="${e.target.result}" 
                   style="max-width: 100%; max-height: 300px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">`;
          };
          reader.readAsDataURL(file);
        } else {
          alert("Please select a photo before submitting.");
          imageInput.click(); // buka lagi kalau kosong
        }
      }
    });
  </script>

<script>
  document.getElementById('actionButton').addEventListener('click', function () {
    document.getElementById('uploadSection').style.display = 'block';
  });

  document.getElementById('imageInput').addEventListener('change', function (event) {
    const preview = document.getElementById('imagePreview');
    preview.innerHTML = ''; // Bersihkan isi sebelumnya
    const file = event.target.files[0];
    
    if (file && file.type.startsWith('image/')) {
      const img = document.createElement('img');
      img.src = URL.createObjectURL(file);
      img.style.maxWidth = '100%';
      img.style.borderRadius = '10px';
      preview.appendChild(img);
    } else {
      preview.innerText = 'Please upload a valid image file.';
    }
  });
</script>

  <!-- WhatsApp Floating Button -->
  <a href="https://wa.me/6289655108829" target="_blank" class="whatsapp-button">
    <i class="bi bi-whatsapp"></i>
  </a>

  <!-- Bootstrap & Icons -->
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css" rel="stylesheet" />
</body>
</html>
