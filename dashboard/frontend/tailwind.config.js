/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Accent — reserved for the prediction, the primary CTA and batting side.
        'cricket-green': '#00C851',
        'accent': {
          DEFAULT: '#22D46B',
          soft: '#0E3B22',
          dim: '#1B8F4B',
        },
        // Secondary accent — the bowling / opposition side.
        'oppo': {
          DEFAULT: '#5B9DFF',
          soft: '#12233F',
          dim: '#3D6FBF',
        },
        'cricket-gold': '#F4C74F',
        'cricket-red': '#FF5C5C',
        // Neutral ramp — deeper, cooler, more separation between layers.
        'ink': {
          950: '#08090B',
          900: '#0C0E11',
          850: '#111318',
          800: '#161920',
          700: '#1E222B',
          600: '#2A2F3A',
          500: '#3A4150',
        },
        'dark-bg': '#08090B',
        'dark-card': '#111318',
        'dark-border': '#232833',
        'dark-text': '#E8EAED',
        'dark-muted': '#8A919E',
      },
      fontFamily: {
        'sans': ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        'sport': ['Inter', 'system-ui', 'sans-serif'],
        'mono': ['"JetBrains Mono"', 'ui-monospace', 'SFMono-Regular', 'monospace'],
      },
      fontSize: {
        'display': ['clamp(2rem, 4.2vw, 3rem)', { lineHeight: '1.05', letterSpacing: '-0.03em' }],
        'score': ['clamp(2.75rem, 6vw, 4rem)', { lineHeight: '0.95', letterSpacing: '-0.04em' }],
      },
      borderRadius: {
        'xl': '0.875rem',
        '2xl': '1.125rem',
      },
      boxShadow: {
        'card': '0 1px 2px rgba(0,0,0,.4), 0 8px 24px -12px rgba(0,0,0,.7)',
        'lift': '0 2px 4px rgba(0,0,0,.4), 0 16px 40px -16px rgba(0,0,0,.8)',
        'accent': '0 0 0 1px rgba(34,212,107,.25), 0 12px 32px -12px rgba(34,212,107,.35)',
      },
      animation: {
        'slide-up': 'slide-up 0.35s cubic-bezier(.16,1,.3,1)',
        'fade-in': 'fade-in 0.4s ease-out',
        'shimmer': 'shimmer 1.6s linear infinite',
      },
      keyframes: {
        'slide-up': {
          '0%': { transform: 'translateY(12px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
        'fade-in': {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        'shimmer': {
          '0%': { backgroundPosition: '-800px 0' },
          '100%': { backgroundPosition: '800px 0' },
        },
      },
    },
  },
  plugins: [],
}
