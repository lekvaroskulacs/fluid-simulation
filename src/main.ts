import { Renderer } from "./renderer";

const canvas: HTMLCanvasElement = <HTMLCanvasElement>document.getElementById("gfx-main");
canvas.setAttribute("width", window.innerWidth.toString());
canvas.setAttribute("height", window.innerHeight.toString());

const renderer = new Renderer(canvas);

renderer.init();

window.addEventListener("resize", () => {
    canvas.setAttribute("width", window.innerWidth.toString());
    canvas.setAttribute("height", window.innerHeight.toString());
    renderer.render();
});


//renderParticles(new Circle(100, 100, 50));


