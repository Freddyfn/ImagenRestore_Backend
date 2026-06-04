import cv2
import numpy as np
import base64
import time

class OpenCVImageProcessor:
    def apply_filters(self, image_data: bytes, method: str, tolerance: float, max_iterations: int, omega: float) -> dict:
        start_time = time.time()
        np_arr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"error": "Invalid image format"}
            
        # Convert to float32 for calculations
        u = img.astype(np.float32)
        f = np.copy(u) # The original noisy image is our source term
        
        # Hyperparameter for diffusion (Poisson)
        alpha = 0.25 
        
        # Red-Black masks for vectorization of GS/SOR
        h, w = u.shape[:2]
        y, x = np.mgrid[0:h-2, 0:w-2]
        mask_red = (x + y) % 2 == 0
        mask_black = (x + y) % 2 == 1
        
        chart_data = []
        final_error = 0.0
        iters_done = 0
        
        # Precompute constants
        coeff_center = 1.0 / (1.0 + 4.0 * alpha)
        
        # Pad arrays to handle boundaries easily (Dirichlet boundary condition = keep borders fixed)
        # We only update the interior: [1:-1, 1:-1]
        for k in range(max_iterations):
            u_old = np.copy(u)
            
            if method == 'jacobi':
                # Vectorized Jacobi
                u_neighbors = u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:]
                u[1:-1, 1:-1] = coeff_center * (f[1:-1, 1:-1] + alpha * u_neighbors)
                
            elif method == 'gauss-seidel' or method == 'sor':
                w_param = 1.0 if method == 'gauss-seidel' else omega
                
                # Update Red
                u_neighbors_red = u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:]
                u[1:-1, 1:-1][mask_red] = (1 - w_param) * u[1:-1, 1:-1][mask_red] + \
                    (w_param * coeff_center) * (f[1:-1, 1:-1][mask_red] + alpha * u_neighbors_red[mask_red])
                
                # Update Black
                u_neighbors_black = u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:]
                u[1:-1, 1:-1][mask_black] = (1 - w_param) * u[1:-1, 1:-1][mask_black] + \
                    (w_param * coeff_center) * (f[1:-1, 1:-1][mask_black] + alpha * u_neighbors_black[mask_black])
                    
            elif method == 'ssor':
                w_param = omega
                
                # Forward Sweep (Red then Black)
                u_neighbors_red = u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:]
                u[1:-1, 1:-1][mask_red] = (1 - w_param) * u[1:-1, 1:-1][mask_red] + \
                    (w_param * coeff_center) * (f[1:-1, 1:-1][mask_red] + alpha * u_neighbors_red[mask_red])
                
                u_neighbors_black = u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:]
                u[1:-1, 1:-1][mask_black] = (1 - w_param) * u[1:-1, 1:-1][mask_black] + \
                    (w_param * coeff_center) * (f[1:-1, 1:-1][mask_black] + alpha * u_neighbors_black[mask_black])
                    
                # Backward Sweep (Black then Red)
                u_neighbors_black = u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:]
                u[1:-1, 1:-1][mask_black] = (1 - w_param) * u[1:-1, 1:-1][mask_black] + \
                    (w_param * coeff_center) * (f[1:-1, 1:-1][mask_black] + alpha * u_neighbors_black[mask_black])
                    
                u_neighbors_red = u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:]
                u[1:-1, 1:-1][mask_red] = (1 - w_param) * u[1:-1, 1:-1][mask_red] + \
                    (w_param * coeff_center) * (f[1:-1, 1:-1][mask_red] + alpha * u_neighbors_red[mask_red])
            
            # Calculate Error (Infinity Norm)
            diff = np.abs(u - u_old)
            current_error = float(np.max(diff))
            
            # Save data points for charting (downsample to avoid huge payloads)
            if k % max(1, max_iterations // 20) == 0 or k == max_iterations - 1 or current_error < tolerance:
                chart_data.append({"iteration": k, "error": current_error})
                
            iters_done = k + 1
            final_error = current_error
            
            if current_error < tolerance:
                break
                
        # Convert back to uint8
        u = np.clip(u, 0, 255).astype(np.uint8)
        
        _, buffer = cv2.imencode('.png', u)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        end_time = time.time()
        elapsed = round(end_time - start_time, 2)
        
        return {
            "restored_image_base64": f"data:image/png;base64,{img_base64}",
            "metrics": {
                "time_seconds": elapsed,
                "final_error": round(final_error, 6),
                "iterations": iters_done,
                "method": method
            },
            "chart_data": chart_data
        }
