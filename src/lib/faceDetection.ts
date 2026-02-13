import { FaceDetector, FilesetResolver } from '@mediapipe/tasks-vision'

import type { DetectionRect } from '../types'
import { clampRect } from './geometry'

const TASKS_WASM_URL =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
const FACE_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite'

let detectorPromise: Promise<FaceDetector> | null = null

type DetectSource = HTMLImageElement | HTMLCanvasElement

async function createFaceDetector(
  delegate: 'GPU' | 'CPU',
): Promise<FaceDetector> {
  const vision = await FilesetResolver.forVisionTasks(TASKS_WASM_URL)

  return FaceDetector.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: FACE_MODEL_URL,
      delegate,
    },
    runningMode: 'IMAGE',
    minDetectionConfidence: 0.45,
  })
}

async function createBestEffortFaceDetector(): Promise<FaceDetector> {
  try {
    return await createFaceDetector('GPU')
  } catch {
    return createFaceDetector('CPU')
  }
}

async function getFaceDetector(): Promise<FaceDetector> {
  if (!detectorPromise) {
    detectorPromise = createBestEffortFaceDetector()
  }

  return detectorPromise
}

async function resetFaceDetectorToCpu(): Promise<FaceDetector> {
  const previousDetector = detectorPromise
    ? await detectorPromise.catch(() => null)
    : null
  const closableDetector = previousDetector as { close?: () => void } | null

  if (closableDetector && typeof closableDetector.close === 'function') {
    closableDetector.close()
  }

  detectorPromise = createFaceDetector('CPU')
  return detectorPromise
}

function detectWithDetector(
  detector: FaceDetector,
  source: DetectSource,
  width: number,
  height: number,
): DetectionRect[] {
  const result = detector.detect(source)

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
        width,
        height,
      )

      return rect
    })
    .filter((rect): rect is DetectionRect => Boolean(rect))
}

function getDetectSourceSize(source: DetectSource): { width: number; height: number } {
  if (source instanceof HTMLImageElement) {
    return {
      width: source.naturalWidth,
      height: source.naturalHeight,
    }
  }

  return {
    width: source.width,
    height: source.height,
  }
}

export async function warmupFaceDetector(): Promise<void> {
  await getFaceDetector()
}

export async function detectFaces(
  source: DetectSource,
): Promise<DetectionRect[]> {
  const { width, height } = getDetectSourceSize(source)
  const detector = await getFaceDetector()

  try {
    return detectWithDetector(detector, source, width, height)
  } catch (firstError) {
    const cpuDetector = await resetFaceDetectorToCpu()

    try {
      return detectWithDetector(cpuDetector, source, width, height)
    } catch (secondError) {
      const firstMessage =
        firstError instanceof Error ? firstError.message : '원인 미상'
      const secondMessage =
        secondError instanceof Error ? secondError.message : '원인 미상'

      throw new Error(
        `얼굴 검출 실패 (재시도 포함): 1차 ${firstMessage} / 2차 ${secondMessage}`,
      )
    }
  }
}
