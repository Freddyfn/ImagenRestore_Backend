import cv2
import numpy as np
import base64
import time

class OpenCVImageProcessor:
    def apply_filters(self, image_data: bytes, method: str, tolerance: float, max_iterations: int) -> dict:
        start_time = time.time()
        np_arr = np.frombuffer(image_data, np.uint8)
        img_1x = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # ============================================================
        # PASO 1: SUPER-RESOLUCIÓN BASE (2x Lanczos-4)
        # Usamos Lanczos-4 puro sin filtros de ruido para no destruir
        # la textura natural de los adoquines.
        # ============================================================
        H_orig, W_orig = img_1x.shape[:2]
        img = cv2.resize(img_1x, (2*W_orig, 2*H_orig), interpolation=cv2.INTER_LANCZOS4)
        
        # ============================================================
        # PIPELINE: Convertir a LAB y procesar SOLO el canal L
        # Esto preserva los colores originales (canales a,b intactos)
        # y aplica la inversión de difusión únicamente a la luminosidad.
        # ============================================================
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        
        # Canal L como float32 para el solver iterativo
        f = l_ch.astype(np.float32)
        H, W = f.shape[:2]
        
        # ============================================================
        # MODELO MATEMÁTICO: Deconvolución Iterativa (Inversión de Difusión)
        #
        # Problema: La imagen f está degradada (borrosa, con ruido suavizado).
        # Modelo:   f = (I + α∇²) u_sharp
        #           La imagen original nítida u sufrió una difusión de calor.
        #
        # Solución: Resolver el sistema lineal (I + α∇²) u = f
        #
        # Discretización (5-point stencil):
        #   (1 - 4α) u_{i,j} + α (u_N + u_S + u_W + u_E) = f_{i,j}
        #
        # Jacobi:
        #   u_{i,j}^{new} = [ f_{i,j} - α(u_N + u_S + u_W + u_E) ] / (1 - 4α)
        #
        # EFECTO: Invierte la difusión. Los bordes se AMPLIFICAN porque
        #         el denominador (1-4α) < 1 escala los valores hacia arriba
        #         en zonas de alto gradiente.
        #
        # ============================================================
        # INVERSIÓN DE DIFUSIÓN EXTREMA REGULARIZADA
        # ============================================================
        alpha = 0.125 # Al límite matemático de la inestabilidad
        diag = 1.0 - 4.0 * alpha
        
        u = np.copy(f)
        u_old = np.empty_like(u)
        
        y, x = np.mgrid[0:H-2, 0:W-2]
        mask_red = (x + y) % 2 == 0
        mask_black = (x + y) % 2 == 1
        
        # Omega óptimo
        cos_max = 0.5 * (np.cos(np.pi / H) + np.cos(np.pi / W))
        rho_jacobi = (4.0 * alpha / abs(diag)) * cos_max
        if rho_jacobi < 1.0:
            omega_opt = 2.0 / (1.0 + np.sqrt(1.0 - rho_jacobi**2))
        else:
            omega_opt = 1.0
            
        chart_data = []
        final_error = 0.0
        
        # Forzamos 150 iteraciones para un afilado BRUTAL
        max_iters = 150 
        
        for k in range(max_iters):
            u_old[:] = u
            
            if method == 'jacobi':
                neighbors = u_old[:-2, 1:-1] + u_old[2:, 1:-1] + u_old[1:-1, :-2] + u_old[1:-1, 2:]
                u[1:-1, 1:-1] = (f[1:-1, 1:-1] - alpha * neighbors) / diag
                
            elif method == 'gauss-seidel' or method == 'sor':
                w = 1.0 if method == 'gauss-seidel' else omega_opt
                
                # Red
                neighbors_r = u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:]
                u_gs_r = (f[1:-1, 1:-1] - alpha * neighbors_r) / diag
                u[1:-1, 1:-1][mask_red] = (1 - w) * u[1:-1, 1:-1][mask_red] + w * u_gs_r[mask_red]
                
                # Black
                neighbors_b = u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:]
                u_gs_b = (f[1:-1, 1:-1] - alpha * neighbors_b) / diag
                u[1:-1, 1:-1][mask_black] = (1 - w) * u[1:-1, 1:-1][mask_black] + w * u_gs_b[mask_black]
                
            elif method == 'ssor':
                w = omega_opt
                
                neighbors = u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:]
                u_gs = (f[1:-1, 1:-1] - alpha * neighbors) / diag
                u[1:-1, 1:-1][mask_red] = (1 - w) * u[1:-1, 1:-1][mask_red] + w * u_gs[mask_red]
                
                neighbors = u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:]
                u_gs = (f[1:-1, 1:-1] - alpha * neighbors) / diag
                u[1:-1, 1:-1][mask_black] = (1 - w) * u[1:-1, 1:-1][mask_black] + w * u_gs[mask_black]
                
                neighbors = u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:]
                u_gs = (f[1:-1, 1:-1] - alpha * neighbors) / diag
                u[1:-1, 1:-1][mask_black] = (1 - w) * u[1:-1, 1:-1][mask_black] + w * u_gs[mask_black]
                
                neighbors = u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:]
                u_gs = (f[1:-1, 1:-1] - alpha * neighbors) / diag
                u[1:-1, 1:-1][mask_red] = (1 - w) * u[1:-1, 1:-1][mask_red] + w * u_gs[mask_red]
            
            # ============================================================
            # REGULARIZACIÓN DE TIKHONOV / FILTRO DE CHOQUE (Cada 10 iters)
            # Previene que la ecuación explote en ruido (checkerboard)
            # mientras permite que los bordes macro sigan afilándose al infinito.
            # ============================================================
            if k > 0 and k % 10 == 0:
                u = cv2.GaussianBlur(u, (3, 3), 0.5)
            
            # Error actual
            diff = np.abs(u - u_old)
            current_error = float(np.max(diff))
            final_error = current_error
            
            chart_data.append({"iteration": k, "error": current_error})
            iters_done = k + 1
        
        # Clip al rango válido de L (0-255 en OpenCV LAB)
        l_final = np.clip(u, 0, 255).astype(np.uint8)
        
        # Reconstruir imagen: L procesado + a,b ORIGINALES intactos
        lab_final = cv2.merge((l_final, a_ch, b_ch))
        result = cv2.cvtColor(lab_final, cv2.COLOR_LAB2BGR)
        
        _, buffer = cv2.imencode('.png', result)
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
