/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { Register } from '../models/Register';
import type { UserResponse } from '../models/UserResponse';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class RegisterService {
    /**
     * @param requestBody
     * @returns UserResponse
     * @throws ApiError
     */
    public static registerCreate(
        requestBody: Register,
    ): CancelablePromise<UserResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/register/',
            body: requestBody,
            mediaType: 'application/json',
        });
    }
}
