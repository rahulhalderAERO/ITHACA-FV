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
\*---------------------------------------------------------------------------*/

Info << "Reading field U\n" << endl;
_U = autoPtr<volVectorField>
     (
         new volVectorField
         (
             IOobject
             (
                 "U",
                 runTime.timeName(),
                 mesh,
                 IOobject::MUST_READ,
                 IOobject::NO_WRITE
             ),
             mesh
         )
     );
volVectorField& U = _U();

volVectorField U0(U);

_U0 = autoPtr<volVectorField>
      (
          new volVectorField(U0)
      );


_S = autoPtr<volVectorField>
     (
         new volVectorField
         (
             IOobject
             (
                 "S",
                 runTime.timeName(),
                 mesh,
                 IOobject::NO_READ,
                 IOobject::AUTO_WRITE
             ),
             U.mesh(),
             dimensionedVector("zero", dimensionSet(0, 1, -2, 0, 0, 0, 0), vector(0,0,0))
         )
     );


Info << "Reading/calculating face flux field phi\n" << endl;

_phi = autoPtr<surfaceScalarField>
       (
           new surfaceScalarField
           (
               IOobject
               (
                   "phi",
                   runTime.timeName(),
                   mesh,
                   IOobject::READ_IF_PRESENT,
                   IOobject::NO_WRITE
               ),
               linearInterpolate(U) & mesh.Sf()
           )
       );
surfaceScalarField& phi = _phi();
surfaceScalarField phi0(phi);

_phi0 = autoPtr<surfaceScalarField>
        (
            new surfaceScalarField(phi0)
        );




_res = autoPtr<volVectorField>
     (
         new volVectorField
         (
             IOobject
             (
                 "res",
                 runTime.timeName(),
                 mesh,
                 IOobject::NO_READ,
                 IOobject::AUTO_WRITE
             ),
             U.mesh(),
             dimensionedVector("zero", dimensionSet(0, 1, -2, 0, 0, 0, 0), vector::zero)
         )
     );



Info << "Reading transportProperties\n" << endl;

_transportProperties = autoPtr<IOdictionary>
                       (
                           new IOdictionary
                           (
                               IOobject
                               (
                                   "transportProperties",
                                   runTime.constant(),
                                   mesh,
                                   IOobject::MUST_READ_IF_MODIFIED,
                                   IOobject::NO_WRITE
                               )
                           )
                       );

Info << "Reading viscosity nu\n" << endl;

_nu = autoPtr<dimensionedScalar>
      (
          new dimensionedScalar
          (
              _transportProperties().lookup("nu")
          )
      );


/*_nu = autoPtr<volScalarField>
      (
          new volScalarField
          (
              IOobject
              (
                  "nu",
                  runTime.timeName(),
                  mesh,
                  IOobject::NO_READ,
                  IOobject::AUTO_WRITE
              ),
              U.mesh(),
              dimensionedScalar("zero", dimensionSet(0, 2, -1, 0, 0, 0, 0), 0.1)
          )
      );
*/





