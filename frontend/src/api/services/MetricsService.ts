/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { Metric } from '../models/Metric';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class MetricsService {
    /**
     * @returns Metric
     * @throws ApiError
     */
    public static metricsList(): CancelablePromise<Array<Metric>> {
        return __request(OpenAPI, {
            method: 'GET',
            url: '/api/metrics/',
        });
    }
}
