/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: 'class',
  content: [
    "./app/templates/**/*.html",    
    "./app/static/src/**/*.js",   
    "./node_modules/flowbite/**/*.js", 
  ],
  daisyui: {
    themes: [],
  },
  theme: {
    extend: {},
  },
  plugins: [
    require("flowbite/plugin"),
    require("daisyui")
  ],
};