import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import TextBox from './TextBox.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <TextBox />
  </StrictMode>,
)
