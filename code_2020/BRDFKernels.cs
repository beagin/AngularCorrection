using System;
using System.Collections.Generic;
using System.Text;

namespace MutiAngularSystemController
{
    class BRDFKernels
    {
        public BRDFKernels()
        {
        }

        #region BRDF ∏˜÷÷∫À
        /// <summary>
        /// Ross ∫Ò≤„
        /// </summary>
        /// <param name="sZenith"></param>
        /// <param name="vZenith"></param>
        /// <param name="rAzimuth"></param>
        /// <returns></returns>
        public double Ross_thick(double sZenith, double vZenith, double rAzimuth)
        {

            double xi = 0.0;
            double cosxi = Math.Cos(sZenith) * Math.Cos(vZenith) + Math.Sin(sZenith) * Math.Sin(vZenith) * Math.Cos(rAzimuth);

            xi = Math.Acos(cosxi);
            double PI = 3.141592654595459;
            double k1 = (PI / 2 - xi) * cosxi + Math.Sin(xi);

            double k = k1 / (Math.Cos(sZenith) + Math.Cos(vZenith)) - PI / 4;
            return (double)k;


        }

        /// <summary>
        /// Ross ±°≤„
        /// </summary>
        /// <param name="sZenith"></param>
        /// <param name="vZenith"></param>
        /// <param name="rAzimuth"></param>
        /// <returns></returns>
        public double Ross_Thin(double sZenith, double vZenith, double rAzimuth)
        {
            double xi = 0.0;
            double cosxi = Math.Cos(sZenith) * Math.Cos(vZenith) + Math.Sin(sZenith) * Math.Sin(vZenith) * Math.Cos(rAzimuth);

            xi = Math.Acos(cosxi);
            double PI = 3.141592654595459;
            double k1 = (PI / 2 - xi) * cosxi + Math.Sin(xi);

            double k = k1 / (Math.Cos(sZenith) + Math.Cos(vZenith)) - PI / 2;
            return (double)k;
        }

        /// <summary>
        /// Li Sparse kernelœ° Ëƒ£–Õ£¨º∏∫Œπ‚—ß∫À
        /// </summary>
        /// <param name="sZenith"></param>
        /// <param name="vZenith"></param>
        /// <param name="rAzimuth"></param>
        /// <returns></returns>
        public double LI_Sparse(double sZenith, double vZenith, double rAzimuth)
        {
            double brratio = 1;
            double hbratio = 2;
            double Pi = 3.14159265459;
            double t1 = brratio * Math.Tan(sZenith);
            double theta_ip = Math.Atan(t1);
            double t2 = brratio * Math.Tan(vZenith);
            double theta_vp = Math.Atan(t2);
            double temp1 = Math.Cos(theta_ip);
            double temp2 = Math.Cos(theta_vp);

            double cosxip = temp1 * temp2 + Math.Sin(theta_ip) * Math.Sin(theta_vp) * Math.Cos(rAzimuth);
            double D1 = t1 * t1 + t2 * t2 - 2 * t1 * t2 * Math.Cos(rAzimuth);
            double D = Math.Sqrt(D1);
            double cost1 = t1 * t2 * Math.Sin(rAzimuth);
            double cost2 = D1 + cost1 * cost1;
            double temp3 = 1 / temp1 + 1 / temp2;
            double cost = hbratio * Math.Sqrt(cost2) / temp3;

            if (cost > 1) cost = 1;

            double t = Math.Acos(cost);
            double O = (t - Math.Sin(t) * cost) * temp3 / Pi;

            double k = O - temp3 + (1 + cosxip) / (2 * temp2);
            return (double)k;
        }
        /// <summary>
        /// Liœ° Ë∫À
        /// </summary>
        /// <param name="sZenith"></param>
        /// <param name="vZenith"></param>
        /// <param name="rAzimuth"></param>
        /// <returns></returns>
        public double LI_SparseR(double sZenith, double vZenith, double rAzimuth)
        {
            double brratio = 1;
            double hbratio = 2;
            double Pi = 3.14159265459;
            double t1 = brratio * Math.Tan(sZenith);
            double theta_ip = Math.Atan(t1);
            double t2 = brratio * Math.Tan(vZenith);
            double theta_vp = Math.Atan(t2);
            double temp1 = Math.Cos(theta_ip);
            double temp2 = Math.Cos(theta_vp);

            double cosxip = temp1 * temp2 + Math.Sin(theta_ip) * Math.Sin(theta_vp) * Math.Cos(rAzimuth);
            double D1 = t1 * t1 + t2 * t2 - 2 * t1 * t2 * Math.Cos(rAzimuth);
            double D = Math.Sqrt(D1);
            double cost1 = t1 * t2 * Math.Sin(rAzimuth);
            double cost2 = D1 + cost1 * cost1;
            double temp3 = 1 / temp1 + 1 / temp2;
            double cost = hbratio * Math.Sqrt(cost2) / temp3;

            if (cost > 1) cost = 1;

            double t = Math.Acos(cost);
            double O = (t - Math.Sin(t) * cost) * temp3 / Pi;

            double k = O - temp3 + (1 + cosxip) / (2 * temp1 * temp2);
            return (double)k;
        }
        /// <summary>
        /// LI÷¬√‹∫À
        /// </summary>
        /// <param name="sZenith"></param>
        /// <param name="vZenith"></param>
        /// <param name="rAzimuth"></param>
        /// <returns></returns>
        public double Li_dense(double sZenith, double vZenith, double rAzimuth)
        {
            double brratio = 1;
            double hbratio = 2;
            double Pi = 3.14159265459;
            double t1 = brratio * Math.Tan(sZenith);
            double theta_ip = Math.Atan(t1);
            double t2 = brratio * Math.Tan(vZenith);
            double theta_vp = Math.Atan(t2);
            double temp1 = Math.Cos(theta_ip);
            double temp2 = Math.Cos(theta_vp);

            double cosxip = temp1 * temp2 + Math.Sin(theta_ip) * Math.Sin(theta_vp) * Math.Cos(rAzimuth);

            double D1 = t1 * t1 + t2 * t2 - 2 * t1 * t2 * Math.Cos(rAzimuth);
            double D = Math.Sqrt(D1);
            double cost1 = t1 * t2 * Math.Sin(rAzimuth);
            double cost2 = D1 + cost1 * cost1;
            double temp3 = 1 / temp1 + 1 / temp2;
            double cost = hbratio * Math.Sqrt(cost2) / temp3;
            if (cost > 1) cost = 1;

            double t = Math.Acos(cost);
            double O = (t - Math.Sin(t) * cost) * temp3 / Pi;

            double k = (1 + cosxip) / (temp2 * (temp3 - O)) - 2;
            return (double)k;
        }
        /// <summary>
        /// LI π˝∂…∫À
        /// </summary>
        /// <param name="sZenith"></param>
        /// <param name="vZenith"></param>
        /// <param name="rAzimuth"></param>
        /// <returns></returns>
        public double Li_Transit(double sZenith, double vZenith, double rAzimuth)
        {
            double brratio = 1;
            double hbratio = 2;
            double Pi = 3.14159265459;
            double t1 = brratio * Math.Tan(sZenith);
            double theta_ip = Math.Atan(t1);
            double t2 = brratio * Math.Tan(vZenith);
            double theta_vp = Math.Atan(t2);
            double temp1 = Math.Cos(theta_ip);
            double temp2 = Math.Cos(theta_vp);

            double cosxip = temp1 * temp2 + Math.Sin(theta_ip) * Math.Sin(theta_vp) * Math.Cos(rAzimuth);

            double D1 = t1 * t1 + t2 * t2 - 2 * t1 * t2 * Math.Cos(rAzimuth);
            double D = Math.Sqrt(D1);
            double cost1 = t1 * t2 * Math.Sin(rAzimuth);
            double cost2 = D1 + cost1 * cost1;
            double temp3 = 1 / temp1 + 1 / temp2;
            double cost = hbratio * Math.Sqrt(cost2) / temp3;
            if (cost > 1) cost = 1;

            double t = Math.Acos(cost);
            double O = (t - Math.Sin(t) * cost) * temp3 / Pi;


            double B = temp3 - O;

            double k = 0;
            if (B > 2)
                k = (1 + cosxip) / (temp2 * B) - 2;
            else
                k = -1.0 * B + (1 + cosxip) / (2 * temp2);

            return (double)k;

        }


        #endregion
    }
}
