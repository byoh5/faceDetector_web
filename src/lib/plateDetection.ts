import type { DetectionRect } from '../types'
import { clampRect } from './geometry'

const OPENCV_JS_URL = 'https://docs.opencv.org/4.10.0/opencv.js'
const CASCADE_FILE = 'haarcascade_russian_plate_number.xml'

type CvNamespace = {
  Mat: new () => {
    delete: () => void
  }
  RectVector: new () => {
    size: () => number
    get: (index: number) => { x: number; y: number; width: number; height: number }
    delete: () => void
  }
  CascadeClassifier: new () => {
    load: (path: string) => boolean
    detectMultiScale: (
      image: unknown,
      objects: unknown,
      scaleFactor?: number,
      minNeighbors?: number,
      flags?: number,
      minSize?: unknown,
      maxSize?: unknown,
    ) => void
    delete: () => void
  }
  Size: new (width: number, height: number) => unknown
  COLOR_RGBA2GRAY: number
  cvtColor: (src: unknown, dst: unknown, code: number, dstCn?: number) => void
  equalizeHist: (src: unknown, dst: unknown) => void
  imread: (source: HTMLCanvasElement) => {
    delete: () => void
  }
  FS_analyzePath: (path: string) => { exists: boolean }
  FS_createDataFile: (
    parent: string,
    name: string,
    data: Uint8Array,
    canRead: boolean,
    canWrite: boolean,
    canOwn: boolean,
  ) => void
  onRuntimeInitialized?: () => void
}

declare global {
  interface Window {
    cv?: CvNamespace
  }
}

let cvPromise: Promise<CvNamespace> | null = null
let openCvScriptPromise: Promise<void> | null = null
let cascadeLoaded = false

async function ensureOpenCvScript(): Promise<void> {
  if (openCvScriptPromise) {
    return openCvScriptPromise
  }

  const existingScript = document.querySelector<HTMLScriptElement>(
    'script[data-opencv-runtime="1"]',
  )

  if (existingScript) {
    openCvScriptPromise = new Promise<void>((resolve, reject) => {
      const startTime = Date.now()

      const waitForCvObject = () => {
        if (window.cv) {
          resolve()
          return
        }

        if (Date.now() - startTime > 12_000) {
          reject(new Error('OpenCV 스크립트 로딩 시간이 초과되었습니다.'))
          return
        }

        window.setTimeout(waitForCvObject, 50)
      }

      waitForCvObject()
    })

    return openCvScriptPromise
  }

  openCvScriptPromise = new Promise<void>((resolve, reject) => {
    const script = document.createElement('script')
    script.src = OPENCV_JS_URL
    script.async = true
    script.defer = true
    script.dataset.opencvRuntime = '1'
    script.onload = () => resolve()
    script.onerror = () => reject(new Error('OpenCV 스크립트를 불러오지 못했습니다.'))
    document.body.appendChild(script)
  })

  return openCvScriptPromise
}

async function ensureCv(): Promise<CvNamespace> {
  if (cvPromise) {
    return cvPromise
  }

  cvPromise = new Promise<CvNamespace>((resolve, reject) => {
    const resolveWhenReady = () => {
      const cv = window.cv

      if (!cv) {
        reject(new Error('OpenCV 객체를 찾지 못했습니다.'))
        return
      }

      if (cv.Mat) {
        resolve(cv)
        return
      }

      const previousRuntimeHandler = cv.onRuntimeInitialized
      cv.onRuntimeInitialized = () => {
        previousRuntimeHandler?.()
        resolve(cv)
      }
    }

    ensureOpenCvScript()
      .then(resolveWhenReady)
      .catch((error) => {
        reject(error)
      })
  })

  return cvPromise
}

async function ensureCascade(cv: CvNamespace): Promise<void> {
  if (cascadeLoaded) {
    return
  }

  const exists = cv.FS_analyzePath(`/${CASCADE_FILE}`).exists

  if (!exists) {
    const response = await fetch(`/${CASCADE_FILE}`)

    if (!response.ok) {
      throw new Error('번호판 cascade 파일을 불러오지 못했습니다.')
    }

    const bytes = new Uint8Array(await response.arrayBuffer())
    cv.FS_createDataFile('/', CASCADE_FILE, bytes, true, false, false)
  }

  cascadeLoaded = true
}

export async function warmupPlateDetector(): Promise<void> {
  const cv = await ensureCv()
  await ensureCascade(cv)
}

export async function detectLicensePlates(
  image: HTMLImageElement,
): Promise<DetectionRect[]> {
  const cv = await ensureCv()
  await ensureCascade(cv)

  const workCanvas = document.createElement('canvas')
  workCanvas.width = image.naturalWidth
  workCanvas.height = image.naturalHeight

  const workContext = workCanvas.getContext('2d')

  if (!workContext) {
    return []
  }

  workContext.drawImage(image, 0, 0)

  const src = cv.imread(workCanvas)
  const gray = new cv.Mat()
  const plates = new cv.RectVector()
  const classifier = new cv.CascadeClassifier()

  try {
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0)
    cv.equalizeHist(gray, gray)

    const loaded = classifier.load(CASCADE_FILE)

    if (!loaded) {
      throw new Error('번호판 분류기 로딩에 실패했습니다.')
    }

    const minSide = Math.max(
      24,
      Math.round(Math.min(image.naturalWidth, image.naturalHeight) * 0.04),
    )

    classifier.detectMultiScale(
      gray,
      plates,
      1.08,
      4,
      0,
      new cv.Size(minSide, Math.max(10, Math.round(minSide * 0.35))),
      new cv.Size(0, 0),
    )

    const result: DetectionRect[] = []

    for (let index = 0; index < plates.size(); index += 1) {
      const plate = plates.get(index)
      const rect = clampRect(
        {
          x: plate.x,
          y: plate.y,
          width: plate.width,
          height: plate.height,
        },
        image.naturalWidth,
        image.naturalHeight,
      )

      if (rect) {
        result.push(rect)
      }
    }

    return result
  } finally {
    src.delete()
    gray.delete()
    plates.delete()
    classifier.delete()
  }
}
