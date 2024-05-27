/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { DemoDocs } from '../models/DemoDocs';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class DemodocsService {
    /**
     * @returns DemoDocs
     * @throws ApiError
     */
    public static demodocsList(): CancelablePromise<Array<DemoDocs>> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/demodocs/',
        });
    }
}
