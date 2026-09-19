import { useEffect, useRef } from "react";
import * as THREE from "three";

function Globe() {
  const containerRef = useRef(null);

  useEffect(() => {
    const container = containerRef.current;

    if (!container) return;

    // ================================
    // SCENE
    // ================================

    const scene = new THREE.Scene();


    // ================================
    // CAMERA
    // ================================

    const camera = new THREE.PerspectiveCamera(
      35,
      container.clientWidth / container.clientHeight,
      0.1,
      100
    );

    camera.position.set(0, 0, 3.2);


    // ================================
    // RENDERER
    // ================================

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
    });

    renderer.setPixelRatio(
      Math.min(window.devicePixelRatio, 2)
    );

    renderer.setSize(
      container.clientWidth,
      container.clientHeight
    );

    renderer.outputColorSpace =
      THREE.SRGBColorSpace;

    renderer.setClearColor(0x000000, 0);

    container.appendChild(renderer.domElement);


    // ================================
    // EARTH SPHERE
    // ================================

    const earthGeometry =
      new THREE.SphereGeometry(
        1,
        128,
        128
      );


    // ================================
    // TEMPORARY EARTH MATERIAL
    // ================================
    //
    // We are intentionally not using
    // earth.png here because your current
    // image is a rendered circular Earth,
    // not a world-map texture.
    //

const textureLoader = new THREE.TextureLoader();

const earthTexture = textureLoader.load(
  "/earth_texture.png"
);

earthTexture.colorSpace =
  THREE.SRGBColorSpace;

const earthMaterial =
  new THREE.MeshPhongMaterial({
    map: earthTexture,

    shininess: 18,

    specular: new THREE.Color(
      0x223344
    ),
  });


    const earth = new THREE.Mesh(
      earthGeometry,
      earthMaterial
    );

    scene.add(earth);


    // ================================
    // GREEN CITY / LAND GLOW
    // ================================

    const glowGeometry =
      new THREE.SphereGeometry(
        1.008,
        128,
        128
      );

    const glowMaterial =
      new THREE.MeshBasicMaterial({
        color: 0x35ff70,

        transparent: true,

        opacity: 0.06,

        side: THREE.BackSide,
      });

    const glow = new THREE.Mesh(
      glowGeometry,
      glowMaterial
    );

    scene.add(glow);


    // ================================
    // ATMOSPHERE
    // ================================

    const atmosphereGeometry =
      new THREE.SphereGeometry(
        1.035,
        128,
        128
      );

    const atmosphereMaterial =
      new THREE.MeshBasicMaterial({
        color: 0x32ff72,

        transparent: true,

        opacity: 0.08,

        side: THREE.BackSide,
      });

    const atmosphere = new THREE.Mesh(
      atmosphereGeometry,
      atmosphereMaterial
    );

    scene.add(atmosphere);


    // ================================
    // LIGHTING
    // ================================

    const ambientLight =
      new THREE.AmbientLight(
        0x8affaa,
        0.35
      );

    scene.add(ambientLight);


    // Main sunlight

    const sun =
      new THREE.DirectionalLight(
        0xffffff,
        2.8
      );

    sun.position.set(
      -3,
      2,
      4
    );

    scene.add(sun);


    // Green light from below/right

    const greenLight =
      new THREE.PointLight(
        0x35ff70,
        3.5,
        5
      );

    greenLight.position.set(
      2,
      -1,
      3
    );

    scene.add(greenLight);


    // ================================
    // ROTATION
    // ================================

    const animate = () => {

      requestAnimationFrame(
        animate
      );

      // Actual 3D rotation
      earth.rotation.y += 0.0018;

      glow.rotation.y =
        earth.rotation.y;

      atmosphere.rotation.y =
        earth.rotation.y;

      renderer.render(
        scene,
        camera
      );
    };

    animate();


    // ================================
    // RESIZE
    // ================================

    const handleResize = () => {

      const width =
        container.clientWidth;

      const height =
        container.clientHeight;

      camera.aspect =
        width / height;

      camera.updateProjectionMatrix();

      renderer.setSize(
        width,
        height
      );
    };

    window.addEventListener(
      "resize",
      handleResize
    );


    // ================================
    // CLEANUP
    // ================================

    return () => {

      window.removeEventListener(
        "resize",
        handleResize
      );

      earthGeometry.dispose();

      earthMaterial.dispose();

      glowGeometry.dispose();

      glowMaterial.dispose();

      atmosphereGeometry.dispose();

      atmosphereMaterial.dispose();

      renderer.dispose();

      if (
        renderer.domElement &&
        container.contains(
          renderer.domElement
        )
      ) {
        container.removeChild(
          renderer.domElement
        );
      }
    };

  }, []);


  return (
    <div
      ref={containerRef}
      className="globe-container"
    />
  );
}

export default Globe;