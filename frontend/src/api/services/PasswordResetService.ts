/* generated using openapi-typescript-codegen -- do not edit */
/* istanbul ignore file */
/* tslint:disable */
/* eslint-disable */
import type { ResetPassword } from '../models/ResetPassword';
import type { UserResponse } from '../models/UserResponse';
import type { CancelablePromise } from '../core/CancelablePromise';
import { OpenAPI } from '../core/OpenAPI';
import { request as __request } from '../core/request';
export class PasswordResetService {
    /**
     * @param requestBody
     * @returns UserResponse
     * @throws ApiError
     */
    public static passwordResetCreate(
        requestBody: ResetPassword,
    ): CancelablePromise<UserResponse> {
        return __request(OpenAPI, {
            method: 'POST',
            url: '/api/password_reset/',
            body: requestBody,
            mediaType: 'application/json',
        });
    }
}
