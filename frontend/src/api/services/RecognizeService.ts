/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { CancelablePromise } from "../core/CancelablePromise";
import { OpenAPI } from "../core/OpenAPI";
import { request as __request } from "../core/request";
import { RecognizeImage } from "../models/RecognizeImage";
export class RecognizeService {
  /**
   * @param formData
   * @returns string
   * @throws ApiError
   */
  public static recognizeCreate(
    formData: RecognizeImage
  ): CancelablePromise<{ text: string }> {
    return __request(OpenAPI, {
      method: "POST",
      url: "/api/recognize/",
      formData: formData,
      mediaType: "multipart/form-data",
    });
  }
}
