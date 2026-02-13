import { FaceDetector, FilesetResolver } from '@mediapipe/tasks-vision'

import type { DetectionRect } from '../types'
import { clampRect } from './geometry'

const TASKS_WASM_URL =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
const FACE_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite'

let detectorPromise: Promise<FaceDetector> | null = null

async function createFaceDetector(): Promise<FaceDetector> {
  const vision = await FilesetResolver.forVisionTasks(TASKS_WASM_URL)

  try {
    return await FaceDetector.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: FACE_MODEL_URL,
        delegate: 'GPU',
      },
      runningMode: 'IMAGE',
      minDetectionConfidence: 0.45,
    })
  } catch {
    return FaceDetector.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: FACE_MODEL_URL,
        delegate: 'CPU',
      },
      runningMode: 'IMAGE',
      minDetectionConfidence: 0.45,
    })
  }
}

async function getFaceDetector(): Promise<FaceDetector> {
  if (!detectorPromise) {
    detectorPromise = createFaceDetector()
  }

  return detectorPromise
}

export async function warmupFaceDetector(): Promise<void> {
  await getFaceDetector()
}

export async function detectFaces(
  image: HTMLImageElement,
): Promise<DetectionRect[]> {
  const detector = await getFaceDetector()
  const result = detector.detect(image)

  return result.detections
    .map((detection) => {
      const box = detection.boundingBox

      if (!box) {
        return null
      }

      const rect = clampRect(
        {
          x: box.originX,
          y: box.originY,
          width: box.width,
          height: box.height,
          score: detection.categories?.[0]?.score,
        },
        image.naturalWidth,
        image.naturalHeight,
      )

      return rect
    })
    .filter((rect): rect is DetectionRect => Boolean(rect))
}
