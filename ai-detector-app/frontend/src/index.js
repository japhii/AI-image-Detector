// index.js
// Entry point for the React application.
// Renders the root <App /> component into the #root div in public/index.html.

import React from "react";
import ReactDOM from "react-dom/client";
import "./App.css";
import App from "./App";

const root = ReactDOM.createRoot(document.getElementById("root"));

root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
