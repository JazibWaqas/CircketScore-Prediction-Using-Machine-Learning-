# 🌐 Using Ngrok to Share Your Cricket Predictor

## **What is Ngrok?**
Ngrok creates a secure public URL for your localhost, allowing others to access your website instantly!

---

## **🔧 UPDATED: Now Works on Phone!**

**Previously:** Ngrok worked on laptop but showed "make sure website is loaded on server 5002" on phone.

**Fixed!** The frontend now automatically uses your ngrok backend URL. Just run `START_WITH_NGROK.bat` and follow the prompts!

See `NGROK_QUICK_FIX.md` for detailed troubleshooting.

---

## **Quick Setup (3 Steps)**

### **Step 1: Sign Up (Optional but Recommended)**
1. Go to https://ngrok.com/
2. Sign up for free account
3. Get your authtoken
4. Run: `ngrok config add-authtoken YOUR_AUTH_TOKEN`

---

### **Step 2: Start Your Applications**

#### **Terminal 1: Start Backend**
```bash
cd dashboard/backend
python app.py
```
**Backend running on:** `http://localhost:5002`

#### **Terminal 2: Start Frontend**
```bash
cd dashboard/frontend
npm start
```
**Frontend running on:** `http://localhost:3000`

---

### **Step 3: Create Ngrok Tunnels**

#### **Terminal 3: Expose Backend**
```bash
ngrok http 5002
```

**Copy the HTTPS URL** (e.g., `https://abc123.ngrok.io`)

#### **Terminal 4: Expose Frontend**
```bash
ngrok http 3000
```

**Copy the HTTPS URL** (e.g., `https://xyz789.ngrok.io`)

---

## **Update Frontend to Use Ngrok Backend**

### **Option 1: Temporary (Environment Variable)**
In Terminal 2 (where frontend runs), stop it (Ctrl+C) and restart with:

**Windows:**
```bash
set REACT_APP_API_URL=https://YOUR-BACKEND-NGROK-URL.ngrok.io/api
npm start
```

**Mac/Linux:**
```bash
REACT_APP_API_URL=https://YOUR-BACKEND-NGROK-URL.ngrok.io/api npm start
```

### **Option 2: Update Config File**

Edit `dashboard/frontend/src/utils/api.js`:
```javascript
const API_BASE_URL = 'https://YOUR-BACKEND-NGROK-URL.ngrok.io/api';
```

---

## **Update Backend CORS**

Edit `dashboard/backend/config.py`:
```python
CORS_ORIGINS = [
    'http://localhost:3000',
    'https://*.ngrok.io',  # Allow all ngrok URLs
    'https://*.ngrok-free.app'  # New ngrok free tier domain
]
```

Restart backend!

---

## **Share Your Website! 🎉**

**Your Public URL:** `https://xyz789.ngrok.io`

Anyone can access it from anywhere in the world!

---

## **Limitations (Free Tier)**

⚠️ **Ngrok Free Tier:**
- Session expires after 2 hours (need to restart)
- URL changes each time you restart
- Bandwidth limits apply
- Shows ngrok warning page before redirecting

✅ **Ngrok Paid ($8/month):**
- Custom domain
- No session limits
- No warning page
- More bandwidth

---

## **Complete Startup Script**

I'll create a helper script for you!

**File: `START_NGROK.bat`** (Windows)
```batch
@echo off
echo Starting Cricket Predictor with Ngrok...
echo.

echo [1/4] Starting Backend...
start "Backend" cmd /k "cd dashboard\backend && python app.py"
timeout /t 5

echo [2/4] Starting Frontend...
start "Frontend" cmd /k "cd dashboard\frontend && npm start"
timeout /t 10

echo [3/4] Starting Ngrok for Backend (Port 5002)...
start "Ngrok Backend" cmd /k "ngrok http 5002"
timeout /t 3

echo [4/4] Starting Ngrok for Frontend (Port 3000)...
start "Ngrok Frontend" cmd /k "ngrok http 3000"

echo.
echo ========================================
echo All services started!
echo ========================================
echo.
echo INSTRUCTIONS:
echo 1. Copy the HTTPS URL from "Ngrok Backend" window
echo 2. Update frontend API URL with that backend URL
echo 3. Copy the HTTPS URL from "Ngrok Frontend" window
echo 4. Share that frontend URL with anyone!
echo.
echo Press any key to open ngrok dashboard...
pause
start https://dashboard.ngrok.com/
```

---

## **Alternative: Easier Single-Command Solution**

Instead of 4 terminals, use ngrok for just the frontend:

1. **Start backend locally:** `python app.py`
2. **Update frontend** to use `http://localhost:5002/api`
3. **Start frontend:** `npm start`
4. **Expose frontend only:** `ngrok http 3000`

**Share:** The ngrok URL for frontend
**Note:** Backend calls will work from shared frontend to your local backend (must keep backend running)

---

## **Troubleshooting**

### **Frontend can't reach backend:**
- Check CORS settings in `config.py`
- Verify API URL in frontend
- Check both ngrok tunnels are running

### **"ERR_NGROK_6022" or similar:**
- Your free session expired (2 hours)
- Restart ngrok tunnels

### **Backend not responding:**
- Check Flask is still running
- Verify port 5002 is not used by another app

---

## **Pro Tip: Keep Everything Local**

For demos, the easiest approach:

1. **Start backend:** `cd dashboard/backend && python app.py`
2. **Start frontend:** `cd dashboard/frontend && npm start`  
3. **Tunnel frontend only:** `ngrok http 3000`
4. **Share the ngrok URL!**

The shared URL will work because:
- Frontend loads in user's browser
- Browser makes API calls directly to backend
- Works as long as your computer is on!

---

**Ready to share your cricket predictor with the world!** 🏏🌍

