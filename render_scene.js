import * as THREE from 'three';
import * as facemesh from '@tensorflow-models/facemesh';
import Stats from 'stats.js';
import * as tf from '@tensorflow/tfjs-core';
import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';
// TODO(annxingyuan): read version from tfjsWasm directly once
// https://github.com/tensorflow/tfjs/pull/2819 is merged.
import {version} from '@tensorflow/tfjs-backend-wasm/dist/version';

import {TRIANGULATION} from './triangulation';
// const stats = new Stats();

let model, ctx, videoWidth, videoHeight, video, canvas,
    scatterGLHasInitialized = false, scatterGL;
let img, imageWidth, imageHeight;

var scene = new THREE.Scene();
var camera = new THREE.OrthographicCamera( -1080, 0 , 0, -1348, 1, 1000 );

// var camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 1000 );

var renderer = new THREE.WebGLRenderer();
renderer.setSize( window.innerWidth, window.innerHeight );
document.body.appendChild( renderer.domElement );

// rotate and position the plane
//plane.rotation.x = -0.5 * Math.PI;
// plane.position.set(150,150,200);
// add the plane to the scene
// scene.add(plane);

var geometry = new THREE.BoxGeometry();
var material = new THREE.MeshBasicMaterial( { color: 0x00ff00 } );
var cube = new THREE.Mesh( geometry, material );
// scene.add( cube );

// camera.position.z = 5;

var animate = function () {
    requestAnimationFrame( animate );

    cube.rotation.x += 0.01;
    cube.rotation.y += 0.01;

    renderer.render( scene, camera );
};

//animate();

function drawPath(ctx, points, closePath) {
    const region = new Path2D();
    region.moveTo(points[0][0], points[0][1]);
    for (let i = 1; i < points.length; i++) {
        const point = points[i];
        region.lineTo(point[0], point[1]);
    }

    if (closePath) {
        region.closePath();
    }
    ctx.stroke(region);
}

let mean_x=0., mean_y=0., mean_z=0.;
var light = new THREE.DirectionalLight(0xffffff, 1);
light.position.setScalar(10);
scene.add(light);

var planeGeometry = new THREE.PlaneGeometry(1080, 1348, 1, 1);
var texture = new THREE.TextureLoader().load( './images/face2.jpg' );
texture.wrapS = THREE.RepeatWrapping;
texture.repeat.x = - 1;
// console.log(texture.image)
var planeMaterial = new THREE.MeshLambertMaterial( { map: texture } );
// console.log('map', planeMaterial.map.image)
//var planeMaterial = new THREE.MeshLambertMaterial({color: 0xffffff});
var plane = new THREE.Mesh(planeGeometry, planeMaterial);
// console.log('map', plane.material.map.image)
plane.receiveShadow = true;
plane.position.set(- 1080/2., -1348/2., -1205);
//plane.position.set(0, 0, 0);
scene.add(plane);

function calculate_means(points) {
    let means = [0., 0., 0.]
    for (let i = 0; i < points.length; i++) {
        means[0] += -points[i][0]
        means[1] += -points[i][1]
        means[2] += -points[i][2]
    }
    means[0] = means[0]/points.length
    means[1] = means[1]/points.length
    means[2] = means[2]/points.length
    return means
}


async function renderPrediction() {
    // stats.begin();
    //console.log(img)
    const predictions = await model.estimateFaces(img);
    //console.log(predictions)
    // ctx.drawImage(
    //    img, 0, 0, imageWidth, imageHeight, 0, 0, canvas.width, canvas.height);

    if (predictions.length > 0) {
        predictions.forEach(prediction => {
            const keypoints = prediction.scaledMesh;
            //const keypoints = prediction.annotations.lipsLowerInner;
            console.log(prediction.annotations)
            console.log(keypoints)
            // let means = calculate_means(keypoints)
            if (true) {
                let overall_points = []
                for (let i = 0; i < TRIANGULATION.length / 3; i++) {
                    overall_points.push(
                        [TRIANGULATION[i * 3], TRIANGULATION[i * 3 + 1],
                        TRIANGULATION[i * 3 + 2]].map(index => keypoints[index]).map(point => ([-point[0], -point[1], -point[2]]))
                    )
                }
                // console.log("points", overall_points.flat())
                var vertices = new Float32Array(overall_points.flat(2));
                //vertices = overall_points.map(index => keypoints[index]).flat();
                // console.log(vertices)
                var geometry = new THREE.BufferGeometry();

                // console.log(vertices)
                geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
                // console.log(geometry)

                // var material = new THREE.MeshBasicMaterial({color: 0x00ff00});
                var material = new THREE.MeshToonMaterial();
                material.side = THREE.DoubleSide;
                var face_mesh = new THREE.Mesh(geometry, material);

                var geo = new THREE.EdgesGeometry(geometry); // or WireframeGeometry( geometry )

                var mat = new THREE.LineBasicMaterial({color: 0xffffff, linewidth: 0.05});

                var wireframe = new THREE.LineSegments(geo, mat);
                //console.log(wireframe)

                scene.add(wireframe);

                camera.position.z = 100;
                // camera.position.z = 605;
                // camera.position.x = 100;
                // camera.position.y = 100;

                // console.log("Means", means)
                // plane.position.set(means[0],means[1],means[2]);
                // scene.add(plane);

                renderer.render( scene, camera );
                scene.remove(wireframe);
                // scene.remove(plane);
                // face_mesh.rotation.x += 0.01;
                // face_mesh.rotation.y += 0.01;

            } else {
                for (let i = 0; i < keypoints.length; i++) {
                    const x = keypoints[i][0];
                    const y = keypoints[i][1];

                    ctx.beginPath();
                    ctx.arc(x, y, 1 /* radius */, 0, 2 * Math.PI);
                    ctx.fill();
                }
            }
        });

        // var geometry = new THREE.BoxBufferGeometry( 1, 1, 1 );
        // var material = new THREE.MeshBasicMaterial( { color: 0xffff00 } );
        // var mesh = new THREE.Mesh( geometry, material );
        // scene.add( mesh );
        // if (renderPointcloud && state.renderPointcloud && scatterGL != null) {
        //     const pointsData = predictions.map(prediction => {
        //         let scaledMesh = prediction.scaledMesh;
        //         return scaledMesh.map(point => ([-point[0], -point[1], -point[2]]));
        //     });
        //
        //     let flattenedPointsData = [];
        //     for (let i = 0; i < pointsData.length; i++) {
        //         flattenedPointsData = flattenedPointsData.concat(pointsData[i]);
        //     }
        //     const dataset = new ScatterGL.Dataset(flattenedPointsData);
        //
        //     if (!scatterGLHasInitialized) {
        //         scatterGL.render(dataset);
        //     } else {
        //         scatterGL.updateDataset(dataset);
        //     }
        //     scatterGLHasInitialized = true;
        // }
    }

    // stats.end();
    requestAnimationFrame(renderPrediction);
};


async function main() {
    await tf.setBackend('webgl');
    //setupDatGui();

    // stats.showPanel(0);  // 0: fps, 1: ms, 2: mb, 3+: custom
    // document.getElementById('main').appendChild(stats.dom);

    img = document.getElementById('image')
    // console.log(img.width)

    // imageWidth = img.width;
    // imageHeight = img.height;
    // canvas = document.getElementById('output');
    // canvas.width = imageWidth;
    // canvas.height = imageHeight;
    // const canvasContainer = document.querySelector('.canvas-wrapper');
    // canvasContainer.style = `width: ${imageWidth}px; height: ${imageHeight}px`;
    //
    // ctx = canvas.getContext('2d');
    // ctx.translate(canvas.width, 0);
    // ctx.scale(-1, 1);
    // ctx.fillStyle = '#32EEDB';
    // ctx.strokeStyle = '#32EEDB';
    // ctx.lineWidth = 0.5;

    model = await facemesh.load({maxFaces: 1});
    renderPrediction();

    // if (renderPointcloud) {
    //     document.querySelector('#scatter-gl-container').style =
    //         `width: ${VIDEO_SIZE}px; height: ${VIDEO_SIZE}px;`;
    //
    //     scatterGL = new ScatterGL(
    //         document.querySelector('#scatter-gl-container'),
    //         {'rotateOnStart': false, 'selectEnabled': false});
    // }
};

main();