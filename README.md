# Virtual Graffiti
![logo](https://github.com/aidanvancil/virtual_graffiti/assets/42700427/31b8720b-67e0-4368-886a-e223dfc00d05)

## Project Abstract
Utilizing lasers to project visuals onto buildings is a captivating and innovative urban artistic expression. ‘Virtual Graffiti’ is about encompassing these methodologies to create a public, scalable, and visual piece of software. By incorporating laser pointers as an input of interactivity the act of ‘drawing’ on buildings will result in endless unique possibilities. With a large-scale projector, a display can be created that uses interactive visuals to merge the landscape of technical and artistic spaces. This application explores the possibilities of augmented reality experiences, real-time audience engagement, and responsive displays. 

The Virtual Graffiti system involves a camera and laptop to track a laser point on a building's exterior, creating graphics from the laser's location and projecting them back onto the surface with a powerful projector. On the backend, is a bastion laptop that hosts the program and computes the real-time data coming from the laser pointer(s) by using open-source libraries and computer vision. 

[Virtual Graffiti Demo](https://youtu.be/wUU7va6CCMU)

## To set up and run Virtual Graffiti, follow these essential equipment and setup steps, applicable to Windows and other operating systems.

### Equipment we used:

- 1 Windows Laptop -  Intel Core 7 - 16GB Memory - HDMI out
- 1 Epson Powerlite 915 Projector - 5000 Lumens
- 1 Canon 90D Video Camera - manual zoom lens
- 1 Libec TH-X Head and Tripod System
- 1 USB Capture Card
- 3 Rechargeable 200mw Lasers  - Green - Red - Purple

### Software Setup:

Before you begin, ensure you have the following installed:

- [Node.js](https://nodejs.org/) (for npm)
- [Python](https://www.python.org/) (3.x recommended)
- [Pip](https://pip.pypa.io/en/stable/installation/) (Python package installer)

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/virtual-graffiti.git
   cd virtual-graffiti
   ```

2. **Optional, but recommended**
    ```
    python3 -m venv <virtual_env_folder_here>
    source <virtual_env_folder_here>/bin/activate
    ```

3. **Install dependencies**
    ```
    pip install -r requirements.txt
    npm install
    ``` 
    (*npm install on directory containing package.json* + *only pip install -r if in virtual environment*)

4. **Running project**
    ```
    python3 manage.py tailwind start
    python3 manage.py runserver
    ```
    ^ These should both be ran in separate terminals.

    If running on WSL (Windows Subsystem Linux), it is recommended to run the following in Powershell for hardware access (open-cv camera).

    If need be, you might need to migrate tailwind changes so run the following in the root directory:

    ```
    python3 manage.py makemigrations
    python3 manage.py migrate
    ```

5. **External packages**

    If external packages or libraries from pip were installed during development run the following (in the virtual environment) and push to GitHub:

    ```
    pip freeze > requirements.txt
    ```

### Location:

Select a building or wall where the laser is visible. For buildings, turn off lights to ease laser tracking. Avoid using powerful lasers on buildings occupied by people.

### Camera Settings:

Keep the camera behind the projector and set the camera to manual mode to disable auto features and fine-tune the color balance for natural lighting. Make sure the entire projection appears in the frame of the camera. Tweak the shutter speed until tracking is optimal, start at a value of 1/60

### App Setup:

Connect your camera to the USB capture card, route the HDMI output from the camera into the capture card’s input, and connect the capture card to your laptop/PC. 

Start the app and press the “calibrate” button

Set the resolution of your camera, avoid using a resolution less than 1024 x 768

When calibrating the HSV value for lasers, in the HSV frame, make sure that the laser appears as a ring where the ring is a clear presentation of the color of the laser you are calibrating.

### Free Draw Mode:

Use your laser(s) to freely draw onto the projected canvas with no time constraints. This mode allows for limitless possibilities for what you may want to create

### Fill Mode:

Choose an image from the image carousel displayed on the left side of the admin page. Once you have selected your image, start fill mode. Use your laser(s) to fill in the canvas with the image that was selected. Once your canvas is 80% filled, the rest of the image will fill in on its own

### Party Mode:

Like Free Draw, use your laser(s) to freely draw onto the projected canvas. This mode introduces a timer that resets the canvas every 30 seconds. This allows for a more collaborative way to use this application as users with a group of more than 3 people are expected to hand off their laser to the next person waiting in the queue. 

### Notes:

To quit a specific mode, press the ‘q’ key on your keyboard

For manual clearing of the canvas. Press the ‘c’ key on your keyboard.

## Current Progress

![settings](https://github.com/aidanvancil/virtual_graffiti/assets/42700427/0e2a59b2-6385-42ef-97a7-87562db8a5c8)
![QR_code](https://github.com/aidanvancil/virtual_graffiti/assets/42700427/624cdc4e-c743-4213-9ef2-b15ec723203b)
![registration](https://github.com/aidanvancil/virtual_graffiti/assets/42700427/eac12238-9b77-4327-b4db-f6a9b384f1d4)
