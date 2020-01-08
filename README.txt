============================
==         README         ==
============================


This program was developed in the course of the 2020 PhD thesis of Florian Schmidt from the University of Münster in Germany.
The title of the thesis was: "Electrophoretic NMR on Correlated Solvent-Ion Migration in Concentrated Electrolytes".

The program was developed for the import of NMR data from either a Bruker Avance or a Bruker Avance III HD, both with a Proton Frequency of 400 MHz.

Two different power sources were used for the eNMR experiments, of which one was build in-house and the other purchased of P&L Sicentific (Sweden).

I gladly put forward the program for display and full or partial reuse in future scientific work (see licence).

If any help is needed with a certain experimental implementation, please let me know.



====Funktionsumfang====

- Import von Bruker-NMR-Daten
    - eNMR Daten (aktuell nur Spannungsabhängig)
        - Schönhoff-Aufbau
        - Pavel-Aufbau
    
- Phasenwinkelanalyse
    - Phasenkorrektur-Analyse
        - Entropieminimierung
        - Spektrenabgleich
        - Vergleich der Phasenkorrigierten Spektren durch Übereinanderlegen

    - Phasenanalyse mittels Fitting
        - Lorentz/Voigt-Peaks
            - superposition von beliebig vielen Peaks
            - Individuelles festsetzen von Parametern
    
    - Regressionsrechnung
        - Berechnung der jeweiligen Mobilitäten aus automatisch bestimmten experimentellen Parametern
    
    - Vergleich der verschiedenen Ergebnisse
        (- einfaches Tool zum Erstellen von Graphen)

- Phasenanalyse mittels 2D FFT --> Mobility ordered Spectroscopy (MOSY)
    - States-Haberkorn-Methode
    - Ermittlung der Mobilitätsachse
    - Plotten von Slices zum Vergleich der Ergebnisse/Peaks
    - Automatische normierung der Intensitäten und Auffindung der Maxima.
    - Durch Signalverlust mit steigender Spannung stellt man eine zu hohe Mobilität fest!!!
