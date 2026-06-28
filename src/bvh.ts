import {vec3, Vector3} from "dom-game-math";
import {Scene} from "./scene.ts";

export namespace BVH {
    export interface Bounds {
        min: Vector3,
        max: Vector3,
    }

    export interface Node {
        bounds: Bounds,
        leftOrOffset: number,
        countOrLeaf: number,
    }

    const maxTriangleCount = 8;
    const maxMeshCount = 2;

    export function buildSceneNodes(scene: Scene) {
        const worldBoundsList = scene.meshInstances.map(instance => {
            const meshBase = scene.meshBases[instance.meshBaseIndex];
            return transformBounds(meshBase.bounds, instance.transform);
        });

        const nodes: Node[] = [];
        nodes.push({} as Node);

        const stack: Array<{ meshInstanceStart: number, meshInstanceCount: number, nodeIndex: number }> = [];
        stack.push({meshInstanceStart: 0, meshInstanceCount: scene.meshInstances.length, nodeIndex: 0});

        while (stack.length > 0) {
            const {meshInstanceStart, meshInstanceCount, nodeIndex} = stack.pop()!;

            // Get world bounds for this node
            const bounds = {min: vec3.new(Infinity), max: vec3.new(-Infinity)};
            for (let i = meshInstanceStart; i < meshInstanceStart + meshInstanceCount; i++) {
                const worldBounds = worldBoundsList[i];
                vec3.min(bounds.min, worldBounds.min, bounds.min);
                vec3.max(bounds.max, worldBounds.max, bounds.max);
            }

            // Add leaf node
            if (meshInstanceCount <= maxMeshCount) {
                nodes[nodeIndex] = createLeafNode({offset: meshInstanceStart, count: meshInstanceCount, bounds});
                continue;
            }

            // Find longest axis
            const extent = vec3.sub(bounds.max, bounds.min);
            let axis = 0;
            if (extent.y > extent.x) axis = 1;
            if (extent.z > vec3.toArray(extent)[axis]) axis = 2;

            // Sort mesh instances by centroid along longest axis
            const slice = scene.meshInstances
                .slice(meshInstanceStart, meshInstanceStart + meshInstanceCount)
                .sort((a, b) => {
                    const aBounds = worldBoundsList[a.meshBaseIndex];
                    const bBounds = worldBoundsList[b.meshBaseIndex];

                    const aCentroid = (vec3.toArray(aBounds.min)[axis] + vec3.toArray(aBounds.max)[axis]) / 2;
                    const bCentroid = (vec3.toArray(bBounds.min)[axis] + vec3.toArray(bBounds.max)[axis]) / 2;
                    return aCentroid - bCentroid;
                });

            for (let i = 0; i < meshInstanceCount; i++) {
                scene.meshInstances[meshInstanceStart + i] = slice[i];
            }

            // Update world bounds list to match sorted mesh instances
            for (let i = meshInstanceStart; i < meshInstanceStart + meshInstanceCount; i++) {
                const instance = scene.meshInstances[i];
                const base = scene.meshBases[instance.meshBaseIndex];
                worldBoundsList[i] = transformBounds(base.bounds, instance.transform);
            }

            const leftNodeIndex = nodes.length;
            const rightNodeIndex = nodes.length + 1;

            nodes[nodeIndex] = createInnerNode({left: leftNodeIndex, bounds});
            nodes.push({} as Node);
            nodes.push({} as Node);

            const leftCount = Math.floor(meshInstanceCount / 2);
            const rightCount = meshInstanceCount - leftCount;

            // Right
            stack.push({
                meshInstanceStart: meshInstanceStart + leftCount,
                meshInstanceCount: rightCount,
                nodeIndex: rightNodeIndex
            });

            // Left
            stack.push({meshInstanceStart, meshInstanceCount: leftCount, nodeIndex: leftNodeIndex});
        }

        scene.sceneNodes.push(...nodes);
    }

    export function buildMeshNodes(meshBase: Scene.MeshBase, scene: Scene) {
        const meshTriangles = scene.triangles.slice(meshBase.triangleStart, meshBase.triangleStart + meshBase.triangleCount);
        const triangleBoundsList = meshTriangles.map(tri => createTriangleBounds(tri));

        meshBase.bounds = {min: vec3.new(Infinity), max: vec3.new(-Infinity)};
        for (const triBounds of triangleBoundsList) {
            vec3.min(meshBase.bounds.min, triBounds.min, meshBase.bounds.min);
            vec3.max(meshBase.bounds.max, triBounds.max, meshBase.bounds.max);
        }

        const nodes: Node[] = [];
        nodes.push({} as Node);

        const stack: Array<{ triangleStart: number, triangleCount: number, nodeIndex: number }> = [];
        stack.push({triangleStart: 0, triangleCount: meshTriangles.length, nodeIndex: 0});

        while (stack.length > 0) {
            const {triangleStart, triangleCount, nodeIndex} = stack.pop()!;

            // Get triangle bounds for this node
            const bounds = {min: vec3.new(Infinity), max: vec3.new(-Infinity)};
            for (let i = triangleStart; i < triangleStart + triangleCount; i++) {
                const triBounds = triangleBoundsList[i];
                vec3.min(bounds.min, triBounds.min, bounds.min);
                vec3.max(bounds.max, triBounds.max, bounds.max);
            }

            // Add leaf node
            if (triangleCount <= maxTriangleCount) {
                nodes[nodeIndex] = createLeafNode({offset: triangleStart, count: triangleCount, bounds});
                continue;
            }

            // Find longest axis
            const extent = vec3.sub(bounds.max, bounds.min);
            let axis = 0;
            if (extent.y > extent.x) axis = 1;
            if (extent.z > vec3.toArray(extent)[axis]) axis = 2;

            // Sort triangles by centroid along longest axis
            const slice = meshTriangles
                .slice(triangleStart, triangleStart + triangleCount)
                .sort((a, b) => {
                    const aCentroid = (vec3.toArray(a.a)[axis] + vec3.toArray(a.b)[axis] + vec3.toArray(a.c)[axis]) / 3;
                    const bCentroid = (vec3.toArray(b.a)[axis] + vec3.toArray(b.b)[axis] + vec3.toArray(b.c)[axis]) / 3;
                    return aCentroid - bCentroid;
                });

            for (let i = 0; i < triangleCount; i++) {
                meshTriangles[triangleStart + i] = slice[i];
            }

            // Update bounds list to match sorted triangles
            for (let i = triangleStart; i < triangleStart + triangleCount; i++) {
                triangleBoundsList[i] = createTriangleBounds(meshTriangles[i]);
            }

            const leftNodeIndex = nodes.length;
            const rightNodeIndex = nodes.length + 1;

            nodes[nodeIndex] = createInnerNode({left: leftNodeIndex, bounds});
            nodes.push({} as Node);
            nodes.push({} as Node);

            const leftCount = Math.floor(triangleCount / 2);
            const rightCount = triangleCount - leftCount;

            // Right
            stack.push({triangleStart: triangleStart + leftCount, triangleCount: rightCount, nodeIndex: rightNodeIndex});

            // Left
            stack.push({triangleStart, triangleCount: leftCount, nodeIndex: leftNodeIndex});
        }

        // Update sorted mesh triangles
        for (let i = 0; i < meshTriangles.length; i++) {
            scene.triangles[meshBase.triangleStart + i] = meshTriangles[i];
        }

        meshBase.nodeStart = scene.meshNodes.length;
        meshBase.nodeCount = nodes.length;
        scene.meshNodes.push(...nodes);
    }
}

function createLeafNode({offset, count, bounds}) {
    return {leftOrOffset: offset, countOrLeaf: count, bounds};
}

function createInnerNode({left, bounds}) {
    return {leftOrOffset: left, countOrLeaf: 0, bounds};
}

function createTriangleBounds(triangle) {
    const min = vec3.min(triangle.a, triangle.b);
    vec3.min(min, triangle.c, min);

    const max = vec3.max(triangle.a, triangle.b);
    vec3.max(max, triangle.c, max);

    return {min, max};
}

export function transformBounds(bounds, transform) {
    const corners = [
        vec3.transform(vec3.new(bounds.min.x, bounds.min.y, bounds.min.z), transform),
        vec3.transform(vec3.new(bounds.min.x, bounds.min.y, bounds.max.z), transform),
        vec3.transform(vec3.new(bounds.min.x, bounds.max.y, bounds.min.z), transform),
        vec3.transform(vec3.new(bounds.min.x, bounds.max.y, bounds.max.z), transform),
        vec3.transform(vec3.new(bounds.max.x, bounds.min.y, bounds.min.z), transform),
        vec3.transform(vec3.new(bounds.max.x, bounds.min.y, bounds.max.z), transform),
        vec3.transform(vec3.new(bounds.max.x, bounds.max.y, bounds.min.z), transform),
        vec3.transform(vec3.new(bounds.max.x, bounds.max.y, bounds.max.z), transform),
    ];

    const transformedBounds = {min: vec3.new(Infinity), max: vec3.new(-Infinity)};
    for (const corner of corners) {
        vec3.min(transformedBounds.min, corner, transformedBounds.min);
        vec3.max(transformedBounds.max, corner, transformedBounds.max);
    }

    return transformedBounds;
}