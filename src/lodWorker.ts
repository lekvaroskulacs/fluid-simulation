import { mat4, vec3 } from "gl-matrix";
import { Plane } from "./plane_mesh";

self.onmessage = (event) => {
    const { cameraPos, gridSize, device } = event.data;
    console.log("sent");
    const lod0 = [];
    const lod1 = [];
    const lod2 = [];

    for (let i = 0; i < gridSize * gridSize; i++) {
        let x = Math.floor(i / gridSize);
        let z = i % gridSize;
        x = (x - gridSize / 2.0) * 2;
        z = (z - gridSize / 2.0) * 2;

        const distance = vec3.distance(vec3.fromValues(x, 0, z), cameraPos);
        let detail = 10;

        if (distance < 10) {
            detail = 256;
            lod0.push(new Plane(1, detail, mat4.translate(mat4.create(), mat4.create(), vec3.fromValues(x, 0, z)), device));
        } else if (distance < 15) {
            detail = 60;
            lod1.push(new Plane(1, detail, mat4.translate(mat4.create(), mat4.create(), vec3.fromValues(x, 0, z)), device));
        } else {
            detail = 10;
            lod2.push(new Plane(1, detail, mat4.translate(mat4.create(), mat4.create(), vec3.fromValues(x, 0, z)), device));
        }
    }

    self.postMessage({ lod0, lod1, lod2 });
};