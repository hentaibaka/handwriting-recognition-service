/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { RecognizeImage } from '../models/RecognizeImage';
import type { RecognizeText } from '../models/RecognizeText';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class RecognizeService {
    /**
     * @param formData
     * @returns RecognizeText
     * @throws ApiError
     */
    public static recognizeCreate(
        formData: RecognizeImage,
    ): CancelablePromise<RecognizeText> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/recognize/',
            formData: formData,
            mediaType: 'multipart/form-data',
        });
    }
}
