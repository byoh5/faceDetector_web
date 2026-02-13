export type RegionKind = 'face' | 'plate' | 'manual'

export type MaskStyle = 'mosaic' | 'blur' | 'black'

export type PreviewMode = 'masked' | 'original'

export interface DetectionRect {
  x: number
  y: number
  width: number
  height: number
  score?: number
}

export interface MaskRegion extends DetectionRect {
  id: string
  kind: RegionKind
  active: boolean
}

export interface ImageMeta {
  name: string
  width: number
  height: number
  sizeBytes?: number
}
