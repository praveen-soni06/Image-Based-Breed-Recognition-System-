 
   // Navigation
   function showSection(id) {
     document.querySelectorAll('.section').forEach(sec => sec.classList.remove('active'));
     document.getElementById(id).classList.add('active');
     
     document.querySelectorAll('.nav-right a').forEach(link => link.classList.remove('active'));
     event.target.classList.add('active');
     window.scrollTo({ top: 0, behavior: "smooth" });
   }
   // Image Preview  
   const imageInput = document.getElementById('imageInput');
   const preview = document.getElementById('preview');
   const result = document.getElementById('result');
   imageInput?.addEventListener('change', function(event) {
     const file = event.target.files[0];
     if (file) {
       const reader = new FileReader();
       reader.onload = function(e) {
         preview.src = e.target.result;
         preview.hidden = false;
       };
       reader.readAsDataURL(file);
     }
   });
   // Breed Prediction  
   function predictBreed() {
     if (!preview.src) {
       result.textContent = "Please upload an image first!";
       result.style.color = "red";
       return;
     }
     const breeds = ["Murrah Buffalo", "Gir Cow", "Sahiwal", "Kankrej", "Mehsana Buffalo"];
     const randomBreed = breeds[Math.floor(Math.random() * breeds.length)];
     result.textContent = "Predicted Breed: " + randomBreed;
     result.style.color = "#2f855a";
   }
   // Signup/Login with localStorage
   const signupForm = document.getElementById('signupForm');
   const loginForm = document.getElementById('loginForm');
   const signupMsg = document.getElementById('signupMsg');
   const loginMsg = document.getElementById('loginMsg');
   signupForm?.addEventListener('submit', function(e) {
     e.preventDefault();
     const name = document.getElementById('signupName').value;
     const email = document.getElementById('signupEmail').value;
     const password = document.getElementById('signupPassword').value;
     localStorage.setItem('user', JSON.stringify({ name, email, password }));
     signupMsg.style.color = "green";
     signupMsg.textContent = "Account created successfully!";
     signupForm.reset();
   });
   loginForm?.addEventListener('submit', function(e) {
     e.preventDefault();
     const email = document.getElementById('loginEmail').value;
     const password = document.getElementById('loginPassword').value;
     const user = JSON.parse(localStorage.getItem('user'));
     if (user && user.email === email && user.password === password) {
       loginMsg.style.color = "green";
       loginMsg.textContent = "Welcome back, " + user.name + "!";
     } else {
       loginMsg.style.color = "red";
       loginMsg.textContent = "invalid credentials!";
     }
     loginForm.reset();
   });