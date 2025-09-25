let signupRole = "User";
let loginRole = "User";

// --- Navigation (pass event from inline) ---
function showSection(id, ev) {
  document.querySelectorAll('.section').forEach(sec => sec.classList.remove('active'));
  const sec = document.getElementById(id);
  if (sec) sec.classList.add('active');

  // update navbar active state
  document.querySelectorAll('.nav-right a').forEach(link => link.classList.remove('active'));
  if (ev && ev.target) {
    ev.target.classList.add('active');
  } else {
    const links = Array.from(document.querySelectorAll('.nav-right a'));
    const found = links.find(l => l.getAttribute('onclick') && l.getAttribute('onclick').includes("'" + id + "'"));
    if (found) found.classList.add('active');
  }
  window.scrollTo({ top: 0, behavior: "smooth" });
}

// --- Role toggle for signup/login ---
function switchRole(section, role, ev) {
  if (section === 'signup') {
    signupRole = role;
    const h = document.getElementById('signupHeading');
    if (h) h.textContent = `${role} Sign Up`;
    document.querySelectorAll('#signup .toggle-btn').forEach(btn => btn.classList.remove('active'));
    if (ev && ev.target) ev.target.classList.add('active');
  } else if (section === 'login') {
    loginRole = role;
    const h = document.getElementById('loginHeading');
    if (h) h.textContent = `${role} Login`;
    document.querySelectorAll('#login .toggle-btn').forEach(btn => btn.classList.remove('active'));
    if (ev && ev.target) ev.target.classList.add('active');
  }
}

// --- Image preview ---
const imageInput = document.getElementById('imageInput');
const preview = document.getElementById('preview');
const result = document.getElementById('result');

imageInput?.addEventListener('change', function (event) {
  const file = event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = function (e) {
    preview.src = e.target.result;
    preview.hidden = false;
  };
  reader.readAsDataURL(file);
});

// --- Breed prediction (calls backend) ---
async function predictBreed() {
  if (!imageInput.files[0]) {
    result.textContent = "⚠ Please upload an image first!";
    result.style.color = "tomato";
    return;
  }

  const file = imageInput.files[0];
  const formData = new FormData();
  formData.append("file", file);

  result.textContent = "⏳ Predicting...";
  result.style.color = "#f59e0b";

  try {
    const response = await fetch("http://127.0.0.1:8000/predict/", {
      method: "POST",
      body: formData
    });

    const data = await response.json();

    if (data.error) {
      result.textContent = "❌ " + data.error;
      result.style.color = "tomato";
    } else {
      result.textContent = `✅ Predicted Breed: ${data.breed} (Confidence: ${(data.confidence*100).toFixed(2)}%)`;
      result.style.color = "#34d399";
    }
  } catch (err) {
    result.textContent = "❌ Error: " + err.message;
    result.style.color = "tomato";
  }
}

// --- Signup handler ---
const signupForm = document.getElementById('signupForm');
const signupMsg = document.getElementById('signupMsg');

signupForm?.addEventListener('submit', function (e) {
  e.preventDefault();
  const name = document.getElementById('signupName').value.trim();
  const email = document.getElementById('signupEmail').value.trim();
  const password = document.getElementById('signupPassword').value;

  if (!email || !password) {
    signupMsg.textContent = "Please fill all required fields.";
    signupMsg.style.color = "tomato";
    return;
  }

  const users = JSON.parse(localStorage.getItem('users') || '[]');
  const exists = users.find(u => u.email === email && u.role === signupRole);
  if (exists) {
    signupMsg.textContent = "Account already exists for this role & email.";
    signupMsg.style.color = "tomato";
    return;
  }

  users.push({ role: signupRole, name, email, password });
  localStorage.setItem('users', JSON.stringify(users));

  signupMsg.textContent = `🎉 Account created as ${signupRole}!`;
  signupMsg.style.color = "#34d399";
  signupForm.reset();

  signupRole = "User";
  const sh = document.getElementById('signupHeading');
  if (sh) sh.textContent = "User Sign Up";
  document.querySelectorAll('#signup .toggle-btn').forEach(btn => btn.classList.remove('active'));
  const userBtn = document.querySelector('#signup .toggle-btn');
  if (userBtn) userBtn.classList.add('active');
});

// --- Login handler ---
const loginForm = document.getElementById('loginForm');
const loginMsg = document.getElementById('loginMsg');

loginForm?.addEventListener('submit', function (e) {
  e.preventDefault();
  const email = document.getElementById('loginEmail').value.trim();
  const password = document.getElementById('loginPassword').value;

  const users = JSON.parse(localStorage.getItem('users') || '[]');
  const user = users.find(u => u.email === email && u.password === password && u.role === loginRole);

  if (user) {
    loginMsg.textContent = `✅ Welcome back, ${user.name} (${user.role})`;
    loginMsg.style.color = "#34d399";
    loginForm.reset();
  } else {
    loginMsg.textContent = "❌ Invalid credentials for selected role.";
    loginMsg.style.color = "tomato";
  }
});
