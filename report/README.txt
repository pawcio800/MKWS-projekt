MKWS - Thermal Simulation of a Showerhead Injector (HTP)
Author: Pawel Polanowski 333489

CONTENTS
  main.tex                  - LaTeX source (Baltic-style template)
  main.pdf                  - compiled report (this is the deliverable PDF)
  injector_thermal_sim.py   - simulation source code
  injector_thermal_3d.gif   - supplementary animation referenced in Sec. 5.1
  figures/                  - figures used in the report

RECOMPILE THE PDF
  Requires 'tectonic' (already installed via Homebrew: brew install tectonic).
  In this folder run:
      tectonic main.tex
  This regenerates main.pdf. Tectonic downloads any needed LaTeX packages
  automatically on the first run (internet connection required once).

REGENERATE THE FIGURES / GIF
  Requires python3 with numpy and matplotlib:
      python3 injector_thermal_sim.py
  This writes the PNG maps and injector_thermal_3d.gif next to the script.
