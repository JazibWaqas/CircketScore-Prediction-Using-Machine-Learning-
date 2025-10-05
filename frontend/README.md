# 🏏 Cricket Score Prediction Frontend

A sleek, dark-themed React application for cricket score prediction with AI-powered insights.

## 🚀 Quick Start

### Prerequisites
- Node.js 16+ 
- npm or yarn

### Installation

1. **Install dependencies:**
```bash
cd frontend
npm install
```

2. **Start the development server:**
```bash
npm start
```

3. **Open your browser:**
Navigate to `http://localhost:3000`

## 🎨 Features

### **Dark Sporty Theme**
- Modern dark interface with cricket-green accents
- Smooth animations and transitions
- Responsive design for all devices

### **Team Creation**
- Select from 172+ teams
- Choose up to 11 players per team
- Real-time player search and filtering
- Team composition analysis

### **Match Context**
- Venue selection with statistics
- Date picker for match scheduling
- Toss winner and decision
- Match type indicators (IPL, T20 World Cup, etc.)

### **AI Predictions**
- Multiple ML models (Random Forest, XGBoost, Linear Regression)
- Confidence scoring
- Winner prediction with margin
- Match intensity analysis

## 🛠️ Tech Stack

- **React 18** - Modern React with hooks
- **Tailwind CSS** - Utility-first styling
- **Framer Motion** - Smooth animations
- **Axios** - API communication
- **React DatePicker** - Date selection
- **Lucide React** - Beautiful icons

## 📱 Responsive Design

- **Desktop**: Full-featured dashboard layout
- **Tablet**: Optimized grid layouts
- **Mobile**: Stacked cards with touch-friendly controls

## 🎯 Key Components

### **TeamSelector**
- Dynamic team selection
- Player management (add/remove)
- Real-time search
- Team composition preview

### **MatchContext**
- Venue statistics display
- Match type toggles
- Context summary

### **PredictionResults**
- Animated score display
- Winner highlighting
- Confidence metrics
- Model information

## 🔧 Configuration

### **API Endpoints**
The app connects to the Flask API at `http://localhost:5000/api`

### **Environment Variables**
Create a `.env` file for configuration:
```
REACT_APP_API_URL=http://localhost:5000/api
REACT_APP_ENVIRONMENT=development
```

## 🎨 Styling

### **Color Palette**
- **Primary**: Cricket Green (#00C851)
- **Secondary**: Cricket Gold (#FFD700)
- **Accent**: Cricket Red (#FF4444)
- **Background**: Dark (#0F0F0F)
- **Cards**: Dark Card (#1A1A1A)

### **Animations**
- Smooth page transitions
- Hover effects
- Loading animations
- Prediction reveals

## 📦 Build for Production

```bash
npm run build
```

This creates an optimized production build in the `build/` folder.

## 🚀 Deployment

The frontend can be deployed to any static hosting service:
- **Vercel**: `vercel --prod`
- **Netlify**: Drag and drop the `build/` folder
- **GitHub Pages**: Use `gh-pages` package

## 🔍 Development

### **Available Scripts**
- `npm start` - Development server
- `npm test` - Run tests
- `npm run build` - Production build
- `npm run eject` - Eject from Create React App

### **Code Structure**
```
src/
├── components/          # React components
│   ├── Header.js       # App header
│   ├── TeamSelector.js # Team creation
│   ├── MatchContext.js # Match settings
│   ├── PredictionResults.js # Results display
│   └── LoadingSpinner.js # Loading states
├── App.js             # Main app component
├── index.js           # React entry point
└── index.css          # Global styles
```

## 🎯 User Experience

### **Intuitive Flow**
1. **Select Teams** - Choose two teams
2. **Add Players** - Select up to 11 players per team
3. **Set Context** - Choose venue and match details
4. **Predict** - Get AI-powered predictions
5. **View Results** - See scores, winner, and confidence

### **Smart Features**
- **Auto-complete** player search
- **Validation** for team completeness
- **Real-time** form updates
- **Responsive** design for all devices

## 🏆 Performance

- **Fast Loading** - Optimized bundle size
- **Smooth Animations** - 60fps transitions
- **Efficient Rendering** - React best practices
- **Mobile Optimized** - Touch-friendly interface

---

**Built with ❤️ for cricket fans and data enthusiasts!**
