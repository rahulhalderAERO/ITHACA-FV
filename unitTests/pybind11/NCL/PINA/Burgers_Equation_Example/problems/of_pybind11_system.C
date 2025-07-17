/*---------------------------------------------------------------------------*\
     ██╗████████╗██╗  ██╗ █████╗  ██████╗ █████╗       ███████╗██╗   ██╗
     ██║╚══██╔══╝██║  ██║██╔══██╗██╔════╝██╔══██╗      ██╔════╝██║   ██║
     ██║   ██║   ███████║███████║██║     ███████║█████╗█████╗  ██║   ██║
     ██║   ██║   ██╔══██║██╔══██║██║     ██╔══██║╚════╝██╔══╝  ╚██╗ ██╔╝
     ██║   ██║   ██║  ██║██║  ██║╚██████╗██║  ██║      ██║      ╚████╔╝
     ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝      ╚═╝       ╚═══╝

 * In real Time Highly Advanced Computational Applications for Finite Volumes
 * Copyright (C) 2017 by the ITHACA-FV authors
-------------------------------------------------------------------------------
License
    This file is part of ITHACA-FV
    ITHACA-FV is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    ITHACA-FV is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with ITHACA-FV. If not, see <http://www.gnu.org/licenses/>.
Description
    Example of a heat transfer Reduction Problem
SourceFiles
    02thermalBlock.C
\*---------------------------------------------------------------------------*/


#include "fvCFD.H"
#include "IOmanip.H"
#include "Time.H"
#include "laplacianProblem.H"
#include "ReducedLaplacian.H"
#include "Burgers.H"
#include "ITHACAPOD.H"
#include "ITHACAutilities.H"
#include <cstddef>
#define _USE_MATH_DEFINES
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include "iostream"
#include "Foam2Eigen.H"

#if PY_VERSION_HEX < 0x03000000
#define MyPyText_AsString PyString_AsString
#else
#define MyPyText_AsString PyUnicode_AsUTF8
#endif

namespace py = pybind11;

class of_pybind11_system : public Burgers
{
public:
    of_pybind11_system(int argc, char* argv[])
    {
        _args = autoPtr<argList>(
            new argList(argc, argv, true, true, /*initialise=*/false));
        argList& args = _args();
#include "createTime.H"
#include "createMesh.H"
#include "createFields.H"
    }
    ~of_pybind11_system() {};

Eigen::MatrixXd getU()
    {
        Eigen::MatrixXd U_eig(Foam2Eigen::field2Eigen(_U()));
        return std::move(U_eig);
    }
Eigen::MatrixXd getS()
    {
        Eigen::MatrixXd U_eig(Foam2Eigen::field2Eigen(_S()));
        return std::move(U_eig);
    }

Eigen::MatrixXd getphi()
    {
        Eigen::MatrixXd phi_eig(Foam2Eigen::field2Eigen(_phi()));
        return std::move(phi_eig);
    }


Eigen::MatrixXd getResidual()
    {
        
        //Info << "_U()" << _U() << endl;
        _res() = (-fvc::div(_phi(), _U()) + fvc::laplacian(_nu(),_U()));
        Eigen::MatrixXd res_eig(Foam2Eigen::field2Eigen(_res()));
        return std::move(res_eig);
    }

Eigen::SparseMatrix<double> get_system_matrix(Eigen::VectorXd& U)//,Eigen::VectorXd& S)
    {
        Foam2Eigen::Eigen2field(_U(), U);
        fvVectorMatrix UEqn(fvm::ddt(_U()) + fvc::div(_phi(), _U()) - fvm::laplacian(_nu(),_U()));
        Eigen::SparseMatrix<double> A;
        Eigen::VectorXd b;
        Foam2Eigen::fvMatrix2Eigen(UEqn, A, b);
        return A;
    }




Eigen::VectorXd get_rhs(Eigen::VectorXd& U)//,Eigen::VectorXd& S)
    {
        Foam2Eigen::Eigen2field(_U(), U);
        fvVectorMatrix UEqn(fvm::ddt(_U()) + fvc::div(_phi(), _U()) - fvm::laplacian(_nu(),_U()));
        Eigen::SparseMatrix<double> A;
        Eigen::VectorXd b;
        Foam2Eigen::fvMatrix2Eigen(UEqn, A, b);
        return b;
    }


void setPrevU()
    {
        _U().storeOldTime();
    }
void updatephi()
    {
        _phi() = linearInterpolate(_U()) & _mesh().Sf();    
    }

void printU()
    {
        Info << _U() << endl;
    }
void setU(Eigen::VectorXd U)
    {
        _U() = Foam2Eigen::Eigen2field(_U(), U);
    }
void setS(Eigen::VectorXd S)
    {
        _S() = Foam2Eigen::Eigen2field(_S(), S);
    }

void printphi()
    {
        Info << _phi() << endl;
    }
void setphi(Eigen::VectorXd phi)
    {
        _phi() = Foam2Eigen::Eigen2field(_phi(), phi);
    }
void solve()
    {
        fvVectorMatrix UEqn(fvm::ddt(_U()) + fvm::div(_phi(), _U()) == fvm::laplacian(_nu(),_U()));
        UEqn.solve();
    }

void exportU(std::string& subFolder, std::string& folder, std::string& fieldname)
    {
        ITHACAstream::exportSolution(_U(), subFolder, folder, fieldname);
    }
void exportphi(std::string& subFolder, std::string& folder, std::string& fieldname)
    {
        ITHACAstream::exportSolution(_phi(), subFolder, folder, fieldname);
    }
/*
void getResidual()
    {
        _res() = fvc::div(_phi(), _U()) - fvc::laplacian(_nu(),_U()) ; 
    }
*/
};

PYBIND11_MODULE(of_pybind11_system, m)
{
    // bindings to Matrix class
    py::class_<of_pybind11_system>(m, "of_pybind11_system")
        .def(py::init([](
                          std::vector<std::string> args) {
            std::vector<char*> cstrs;
            cstrs.reserve(args.size());
            for (auto& s : args)
                cstrs.push_back(const_cast<char*>(s.c_str()));
            return new of_pybind11_system(cstrs.size(), cstrs.data());
        }),
            py::arg("args") = std::vector<std::string> { "." })        
        .def("getU", &of_pybind11_system::getU, py::return_value_policy::reference_internal)
        .def("getS", &of_pybind11_system::getS, py::return_value_policy::reference_internal)
        .def("setU", &of_pybind11_system::setU, py::return_value_policy::reference_internal)
        .def("setS", &of_pybind11_system::setS, py::return_value_policy::reference_internal)
        .def("getphi", &of_pybind11_system::getphi, py::return_value_policy::reference_internal)
        .def("setphi", &of_pybind11_system::setphi, py::return_value_policy::reference_internal)
        .def("printU", &of_pybind11_system::printU)
        .def("printphi", &of_pybind11_system::printphi)
        .def("get_system_matrix", &of_pybind11_system::get_system_matrix)
        .def("get_rhs", &of_pybind11_system::get_rhs)
        .def("exportU", &of_pybind11_system::exportU)
        .def("exportphi", &of_pybind11_system::exportphi)
        .def("setPrevU", &of_pybind11_system::setPrevU)
        .def("updatephi", &of_pybind11_system::updatephi)
        .def("getResidual", &of_pybind11_system::getResidual)
        .def("solve", &of_pybind11_system::solve);
}

/*

        .def("getT", &of_pybind11_system::getT, py::return_value_policy::reference_internal)
        .def("setT", &of_pybind11_system::setT, py::return_value_policy::reference_internal)
        
        .def("getS", &of_pybind11_system::getS, py::return_value_policy::reference_internal)
        .def("printS", &of_pybind11_system::printS)
        .def("getResidual", &of_pybind11_system::get_residual)
        .def("printMatrix", &of_pybind11_system::printMatrix)
        .def("get_rhs", &of_pybind11_system::get_rhs)
        .def("get_system_matrix", &of_pybind11_system::get_system_matrix)
        .def("solve", &of_pybind11_system::solve)
        .def("exportT", &of_pybind11_system::exportT);


*/ 
