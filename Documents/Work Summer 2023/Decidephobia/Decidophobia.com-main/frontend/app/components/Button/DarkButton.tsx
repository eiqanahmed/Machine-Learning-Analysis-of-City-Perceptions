import React, { useEffect } from "react";
import { atom, useAtom } from "jotai";

export const darkMode = atom(
  typeof window !== "undefined"
    ? localStorage.getItem("theme") || "light"
    : "light"
);

function DarkButton() {
  const [isDarkMode, setIsDarkMode] = useAtom(darkMode);

  useEffect(() => {
    const initialTheme = localStorage.getItem("theme") || "";
    document.documentElement.setAttribute("data-theme", initialTheme);
    console.log("Initial theme applied:", initialTheme);
  }, []);

  const toggleTheme = () => {
    const newTheme = isDarkMode !== "dark" ? "dark" : "";
    setIsDarkMode(newTheme);

    localStorage.setItem("theme", newTheme);
    document.documentElement.setAttribute("data-theme", newTheme);

    console.log("Theme toggled to:", newTheme);
  };

  return (
    <button
      onClick={toggleTheme}
      className="p-2 bg-primary rounded flex items-center h-full text-xl font-semibold no-underline"
    >
      Dark
    </button>
  );
}

export default DarkButton;
