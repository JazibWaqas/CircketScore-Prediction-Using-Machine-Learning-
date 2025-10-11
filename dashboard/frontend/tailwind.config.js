/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'cricket-green': '#00C851',
        'cricket-gold': '#FFD700',
        'cricket-red': '#FF4444',
        'dark-bg': '#0F0F0F',
        'dark-card': '#1A1A1A',
        'dark-border': '#2D2D2D',
        'dark-text': '#E5E5E5',
        'dark-muted': '#8B8B8B',
      },
      fontFamily: {
        'sport': ['Inter', 'system-ui', 'sans-serif'],
      },
      animation: {
        'pulse-glow': 'pulse-glow 2s ease-in-out infinite alternate',
        'slide-up': 'slide-up 0.3s ease-out',
        'fade-in': 'fade-in 0.5s ease-in',
        'bounce-subtle': 'bounce-subtle 0.6s ease-in-out',
      },
      keyframes: {
        'pulse-glow': {
          '0%': { boxShadow: '0 0 5px #00C851' },
          '100%': { boxShadow: '0 0 20px #00C851, 0 0 30px #00C851' },
        },
        'slide-up': {
          '0%': { transform: 'translateY(20px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        'fade-in': {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        'bounce-subtle': {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-5px)' },
        },
      },
    },
  },
  plugins: [],
}
