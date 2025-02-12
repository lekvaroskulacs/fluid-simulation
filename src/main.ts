const output_label : HTMLElement = <HTMLElement>document.getElementById("compatibility-check");

if (navigator.gpu) {
    output_label.innerHTML = "WebGPU supported";
}
else {
    output_label.innerHTML = "WebGPU NOT supported by this browser\nSupported browser: Google Chrome";
}