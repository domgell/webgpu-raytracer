<img height="250" src="https://github.com/user-attachments/assets/c08e8ade-b813-4930-890f-ba4eaa4f3d32" />
<img height="250" src="https://github.com/user-attachments/assets/7e274f04-5bd9-40df-b31f-614d572a582f" />
<img height="250" src="https://github.com/user-attachments/assets/9a67facc-e3ff-4e6b-9b01-7620ca157b6e" />

# About
This is a WebGPU, compute shader based implementation of ray tracing along with a few acceleration techniques such as BVH, NEE and a bilateral denoising filter.

# Usage
**A web based version is hosted at: [domgell.github.io](https://domgell.github.io/webgpu-raytracer/)**

Note, this requires your device supports WebGPU and a relatively modern dedicated GPU for optimal performance. 
Check [Implementation-Status](https://github.com/gpuweb/gpuweb/wiki/Implementation-Status) for more information on WebGPU support.

## Running locally
Requires Node.js and npm are installed.

(In the project root directory) Install dependencies:
```
npm i
```
Start local development server:
```
npm run dev
```
Open `http://localhost:5173/` in the browser.
