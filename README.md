# Hackathon 2024 - Abgabe der Gruppe Vitellium

Teammitglieder:

    - Felix Dallmann
    - Louis Bordon


## Beschreibung

Wir haben die Aufgabe grundsätzlich in zwei Teile aufgespaltet: Zuerst müssen die Kanten und Löcher des Teils erkannt werden, dann kann eine Position für den Gripper gefunden werden.

Für den ersten Teil haben wir das Bild des Teils in den HSV-Farbraum aufgespaltet. Auf den Kanal mit den schärfsten Kanten haben wir für die Kantenerkennung den Canny-Algorithmus angewendet, um das Bild in verschiedene, durch Kanten abgetrennte Bereiche aufzuspalten. Grundsätzlich nehmen wir vom größten Bereich an, dass er das Teil ist. Der Rest wird als Loch klassifiziert.

Für den zweiten Teil haben wir eine Art Gradient-Descent-Algorithmus entwickelt. Mehrere zufällige Startpunkte und -winkel werden so lange in die Richtung verschoben, in der die Anzahl der überlappenden Pixel des Grippers und der Löcher sinkt. Anschließend werden alle gefundenen Lösungen mit einem einfachen Algorithmus so weit wie möglich in die Mitte verschoben. Die beste Lösung wird ausgegeben.

## How to Run

Unser Code funktioniert mit dem vorgegebenen Interface, die main()-Funktion haben wir nicht verändert.

## Sonstiges

Wir haben die meiste Zeit in einem privaten Github-Repository gearbeitet, damit andere sich nicht von unserer Lösung inspirieren lassen können (auch wenn sie nicht besonders originell ist). Deswegen sieht es so aus, als hätten wir alles heute erst irgendwo kopiert.

Auch wenn wir nicht gewinnen: Vielen Dank für die Organisation! Wir hatten viel Spaß bei der Sach und haben viel gelernt.
