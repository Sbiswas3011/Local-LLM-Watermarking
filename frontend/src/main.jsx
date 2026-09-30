import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import TextBox from './TextBox.jsx'
import Settings from './Settings.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    {/* <App /> */}
    {/* <Settings /> */}
    <TextBox />
  </StrictMode>,
)
