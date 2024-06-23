/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        accent: '#00538A',
        error: '#F4547A',
        success: '#E1F5E3',
        green: '#0AAF1A',
        'additional-color-1': '#E6EEF4',
        'additional-color-2': '#303030',
        'additional-color-3': '#26203B',
        'additional-color-4': '#493939',
        'additional-color-5': '#4E4E4E',
        'additional-color-6': '#162229',
        'additional-color-7': '#9C9AA5'
      },
      boxShadow: {
        'card': '0 0 25px rgba(48, 48, 48, 0.08)',
      },
    },
  },
  // eslint-disable-next-line no-undef
  plugins: [require("tailwindcss-animate"),],
}

