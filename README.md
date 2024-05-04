# Virtual Graffiti
![logo](https://github.com/aidanvancil/virtual_graffiti/assets/42700427/31b8720b-67e0-4368-886a-e223dfc00d05)

Welcome to Virtual Graffiti! This project allows you to create amazing digital and displayable artwork with ease. Follow the instructions below to get started (for Team 33 [Aidan Vancil, Foster Schmidt, Moises Moreno]).

## Prerequisites

Before you begin, ensure you have the following installed:

- [Node.js](https://nodejs.org/) (for npm)
- [Python](https://www.python.org/) (3.x recommended)
- [Pip](https://pip.pypa.io/en/stable/installation/) (Python package installer)

## Setup

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
    ^ These should both be ran is separate terminals.

    If running on WSL (Windows Subsystem Linux), it is recommended to run the following in Powershell for hardware access (open-cv camera).

    If need be, you might need to migrate tailwind changes so run the following in the root directory:

    ```
    python3 manage.py makemigrations
    python3 manage.py migrate
    ```

5. **External packages**

    If external packages or libraries from pip were installed during development run the following (in the virtual environment) and push to github:

    ```
    pip freeze > requirements.txt
    ```

## Current Progress

![settings](https://github.com/aidanvancil/virtual_graffiti/assets/42700427/0e2a59b2-6385-42ef-97a7-87562db8a5c8)
![QR_code](https://github.com/aidanvancil/virtual_graffiti/assets/42700427/624cdc4e-c743-4213-9ef2-b15ec723203b)
![registration](https://github.com/aidanvancil/virtual_graffiti/assets/42700427/eac12238-9b77-4327-b4db-f6a9b384f1d4)

## Future Changes
   Our team felt that with the limitations of OpenCV as a drawing application that the next most logical step would be to use a graphics rendering library for our drawing. The choice was made to use OpenGL, and progress was made in creating antialiased lines that are drawn with triangles for better varying degrees of widths for line sizes. Progress can be found [here](/virtual_graffiti/open_gl/) .
