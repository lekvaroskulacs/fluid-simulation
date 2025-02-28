import { Renderer } from "./renderer";

const canvas: HTMLCanvasElement = <HTMLCanvasElement>document.getElementById("gfx-main");
canvas.setAttribute("width", window.innerWidth.toString());
canvas.setAttribute("height", window.innerHeight.toString());

const renderer = new Renderer(canvas);

renderer.init();

window.addEventListener("resize", () => {
    const form = <HTMLElement> document.getElementById("form");
    const formHeight = form.offsetHeight;
    canvas.setAttribute("width", window.innerWidth.toString());
    canvas.setAttribute("height", (window.innerHeight - formHeight).toString());
    renderer.render();
});


(<HTMLInputElement> document.getElementById("amplitude")).value = localStorage.getItem("amplitude") ?? "1";
(<HTMLInputElement> document.getElementById("amplitudeMultiplier")).value = localStorage.getItem("amplitudeMultiplier") ?? "1";
(<HTMLInputElement> document.getElementById("frequency")).value = localStorage.getItem("frequency") ?? "1";
(<HTMLInputElement> document.getElementById("frequencyMultiplier")).value = localStorage.getItem("frequencyMultiplier") ?? "1";
(<HTMLInputElement> document.getElementById("basePhase")).value = localStorage.getItem("basePhase") ?? "0";
(<HTMLInputElement> document.getElementById("baseSpeed")).value = localStorage.getItem("baseSpeed") ?? "1";
(<HTMLInputElement> document.getElementById("maxWaves")).value = localStorage.getItem("maxWaves") ?? "1";
(<HTMLInputElement> document.getElementById("sunPosition")).value = localStorage.getItem("sunPosition") ?? "1";
//renderParticles(new Circle(100, 100, 50));


