/******************************************************************************
 *   Copyright (C) 2006-2024 by the GIMLi development team                    *
 *   Carsten Rücker carsten@resistivity.net                                   *
 *   Thomas Günther thomas@resistivity.net                                    *
 *                                                                            *
 *   Licensed under the Apache License, Version 2.0 (the "License");          *
 *   you may not use this file except in compliance with the License.         *
 *   You may obtain a copy of the License at                                  *
 *                                                                            *
 *       http://www.apache.org/licenses/LICENSE-2.0                           *
 *                                                                            *
 *   Unless required by applicable law or agreed to in writing, software      *
 *   distributed under the License is distributed on an "AS IS" BASIS,        *
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. *
 *   See the License for the specific language governing permissions and      *
 *   limitations under the License.                                           *
 *                                                                            *
 ******************************************************************************/

#include "ttdijkstramodelling.h"

#define NEWREGION 33333

#include "blockmatrix.h"
#include "datacontainer.h"
#include "elementmatrix.h"
#include "pos.h"
#include "mesh.h"
#include "meshgenerators.h"
#include "meshentities.h"
#include "numericbase.h"
#include "regionManager.h"
#include "sparsematrix.h"
#include "calculateMultiThread.h"

#include <vector>
#include <queue>
#include <map>
#include <omp.h>


namespace GIMLI {

Dijkstra::Dijkstra()
: _root(std::numeric_limits<Index>::max()){

}

Dijkstra::Dijkstra(const Graph & graph)
: graph_(graph), _root(std::numeric_limits<Index>::max()){
    pathMatrix_.resize(graph.size());
}

double Dijkstra::distance(Index root, Index node) {
    if (this->_root != root){
        this->setStartNode(root);
    }
    return distance(node);
}

double Dijkstra::distance(Index node) {
    return distances_[node].time();
}

RVector Dijkstra::distances(Index root) {
    this->setStartNode(root);
    return distances();
}

RVector Dijkstra::distances() const {
    RVector ret(0);
    for (auto const & it: distances_){
        ret.push_back(it.second.time());
    }
    return ret;
}

void Dijkstra::setGraph(const Graph & graph) {
    graph_ = graph;
    pathMatrix_.clear();
    pathMatrix_.resize(graph.size());
}

void Dijkstra::setStartNode(Index startNode) {
    distances_.clear();
    this->_root = startNode;
    std::priority_queue< DistancePair_,
                         std::vector< DistancePair_ >,
                         ComparePairsClass_< DistancePair_ > > priQueue;

    Edge_ e(startNode, startNode);
    priQueue.push(DistancePair_(0.0, e));
    DistancePair_ dummy;

    while (!priQueue.empty()) {
        dummy = priQueue.top();

        double distance = dummy.first;
        Index node = dummy.second.end;
        priQueue.pop();

        if (distances_.count(node) == 0) {
            //distances_[node] = distance;

            distances_[node] = GraphDistInfo(distance, 0);

            if ((Index)pathMatrix_.size() <= node){
                std::cout << "startNodeID:" << startNode << " NodeID:" << node << std::endl;
                throwError(WHERE_AM_I + " Warning! Dijkstra graph invalid" );
            }
            pathMatrix_[node] = Edge_(dummy.second);

            NodeDistMap::iterator start = graph_[node].begin();
            NodeDistMap::iterator stop = graph_[node].end();

            for (; start != stop; ++start) {
                e.start = node;
                e.end = (*start).first;
                priQueue.push(DistancePair_(distance + (*start).second.time(), e));
            }
        }
    }
}

void Dijkstra::shortestPathTo(Index node, IndexArray & rway) const{
    IndexArray way;
    Index parentNode = -1, endNode = node;

    while (parentNode != _root) {
        parentNode = pathMatrix_[endNode].start;
        way.push_back(endNode);
        endNode = pathMatrix_[endNode].start;
    }
    way.push_back(_root);

    // collect reverse way
    rway.clear();
    rway.resize(way.size());
    for (Index i = 0; i < way.size(); i ++) rway[i] = way[way.size() - i - 1];
}

IndexArray Dijkstra::shortestPathTo(Index node) const {
    IndexArray rway;
    this->shortestPathTo(node, rway);
    return rway;
}

IndexArray Dijkstra::shortestPath(Index start, Index end){
    if (this->_root != start){
        this->setStartNode(start);
    }
    return this->shortestPathTo(end);
}

//    RVector TravelTimeDijkstraModelling::operator () (const RVector & slowness, double background) {
//        return response(slowness, background);
//    }

TravelTimeDijkstraModelling::TravelTimeDijkstraModelling(bool verbose)
    : ModellingBase(verbose), background_(1e16){
    this->initJacobian();
}

TravelTimeDijkstraModelling::TravelTimeDijkstraModelling(Mesh & mesh,
                                                         DataContainer & dataContainer,
                                                         bool verbose)
    : ModellingBase(dataContainer, verbose), background_(1e16) {

    this->setMesh(mesh);
    this->initJacobian();
}

RVector TravelTimeDijkstraModelling::createDefaultStartModel() {
    return RVector(this->regionManager().parameterCount(), findMedianSlowness());
}

bool V_ = false;

void fillGraph_(Graph & graph, const Node & a, const Node & b, double slowness, SIndex leftID){
    if (a.id() == b.id()) return;

    double dist = a.pos().distance(b.pos());

    // ensure connection between 3d boundaries
    dist = max(1e-8, dist);

    double newTime = dist * slowness;
    double oldTime = graph[a.id()][b.id()].time();

    // if (V_){
    //     __MS("a:" << a.id() << " b:"  << b.id() << " L:" << leftID << " t:" << " " << newTime << " " << oldTime)
    // }

    if (oldTime > 0.0) {
        newTime = std::min(newTime, oldTime);

        // way pair already exist so set time to min and add leftID

        NodeDistMap::iterator ita(graph[a.id()].find(b.id()));
        ita->second.cellIDs().insert(leftID);
        ita->second.setTime(newTime);

        NodeDistMap::iterator itb(graph[b.id()].find(a.id()));
        itb->second.cellIDs().insert(leftID);
        itb->second.setTime(newTime);

    } else {
        // first time fill
        graph[a.id()][b.id()] = GraphDistInfo(newTime, dist, leftID);
        graph[b.id()][a.id()] = GraphDistInfo(newTime, dist, leftID);
    }
}

void fillGraph_(Graph & graph, Cell & c, double slowness){

    std::vector< Node * > ni(c.nodes());

    for (Index i(0); i < c.boundaryCount(); i++){
        Boundary *b = c.boundary(i);
        if (b){
            for (auto & n : b->secondaryNodes()){
                ni.push_back(n);
            }
        } else {
            log(Critical, "No boundary found.");
        }
    }

    for (auto & n : c.secondaryNodes()){
        ni.push_back(n);
    }

    for (Index j = 0; j < ni.size()-1; j ++) {
        for (Index k = j + 1; k < ni.size(); k ++) {
            fillGraph_(graph, *ni[j], *ni[k], slowness, c.id());
        }
    }
}

Graph TravelTimeDijkstraModelling::createGraph(const RVector & slownessPerCell) const {
    Graph graph;
    mesh_->createNeighborInfos();

    for (Index i = 0; i < mesh_->cellCount(); i ++) {
        Cell & c = mesh_->cell(i);
        fillGraph_(graph, c, slownessPerCell[c.id()]);
    }

    if (graph.size() < mesh_->nodeCount()){
        std::cerr << WHERE_AM_I <<
                " there seems to be unassigned nodes within the mesh. Dijkstra Path will be maybe invalid."
                 << graph.size() << " < " << mesh_->nodeCount() << std::endl;
    }
    return graph;
}

double TravelTimeDijkstraModelling::findMedianSlowness() const {
    return median(getApparentSlowness());
}

RVector TravelTimeDijkstraModelling::getApparentSlowness() const {
    if (!dataContainer_) return 0.0;

    Index nData = dataContainer_->size();
    SIndex s = 0, g = 0;
    double edgeLength = 0.0;
    RVector apparentSlowness(nData);

    for (Index dataIdx = 0; dataIdx < nData; dataIdx ++) {
        s = (SIndex)(*dataContainer_)("s")[dataIdx];
        g = (SIndex)(*dataContainer_)("g")[dataIdx];
        if (s == g){
            __MS(WHERE_AM_I + ": shot point equals geophon point. " +
                  "This lead to an invalid apparent slowness. " +
                  str(s) + "==" + str(g))
            throwError("Aborting" );
        }
        edgeLength = dataContainer_->sensorPosition(s).distance(dataContainer_->sensorPosition(g));
        apparentSlowness[dataIdx] = dataContainer_->get("t")[dataIdx] / edgeLength;
    }
    return apparentSlowness;
}

RVector TravelTimeDijkstraModelling::createGradientModel(double lBound,
                                                         double uBound){
    if (verbose_) std::cout << "Creating Gradient model ..." << std::endl;

    RVector appSlowness(getApparentSlowness());

    double smi = median(appSlowness);
    if (smi < lBound) smi = lBound * 1.1;

    double sma = max(appSlowness) / 2.0;
    if (uBound > 0.0 && sma > uBound) sma = uBound * 0.9;

    Index nModel = regionManager().parameterCount();

    RVector zmid(nModel);
    Mesh paraDomain(regionManager().paraDomain());

    int dim = paraDomain.dim() - 1;

    for (Index i = 0; i < paraDomain.cellCount() ; i++) {
        zmid[i] = paraDomain.cell(i).center()[dim];
    }

    double zmi = min(zmid);
    double zma = max(zmid);

    RVector gradModel(nModel);

    for (Index i = 0; i < gradModel.size(); i++) {
        gradModel[i] = smi * std::exp((zmid[i] - zmi) / (zma - zmi) * std::log(sma / smi));
    }

    return gradModel;
}

void TravelTimeDijkstraModelling::updateMeshDependency_(){
    if (verbose_) std::cout << "... looking for shot and receiver positions." << std::endl;

    if (!dataContainer_){
        throwError("We have no dataContainer defined");
    }
    RVector shots(unique(sort((*dataContainer_)("s"))));

    if (shots.size() == 0){
        throwError("There are no shot positions in the dataContainer.");
    }
    shotNodeId_.resize(shots.size()) ;
    shotsInv_.clear();

    if (shots[0] < 0){
        throwError("There are shots index lower then 0.");
    }

    for (Index i = 0; i < shots.size(); i ++){
        shotNodeId_[i] = mesh_->findNearestNode(dataContainer_->sensorPosition((Index)shots[i]));
        if (mesh_->node(shotNodeId_[i]).cellSet().size() == 0 &&
            mesh_->node(shotNodeId_[i]).id() < (int)mesh_->nodeCount()){
            __MS("no cells found for node "  << shotNodeId_[i])
        }
        shotsInv_[Index(shots[i])] = i;
    }

    RVector receiver(unique(sort((*dataContainer_)("g"))));

    receNodeId_.resize(receiver.size()) ;
    receiInv_.clear();

    for (Index i = 0; i < receiver.size(); i ++){
        receNodeId_[i] = mesh_->findNearestNode(dataContainer_->sensorPosition(Index(receiver[i])));
        receiInv_[Index(receiver[i])] = i;
    }
}

class CreateDijkstraDistMT : public GIMLI::BaseCalcMT{
public:
    CreateDijkstraDistMT(RMatrix & dists,
                        const Dijkstra        & dijk,
                        const IndexArray      & shotNodes,
                        const IndexArray      & recNodes,
                        bool verbose)
    : BaseCalcMT(verbose), _dists(&dists), _dijkstra(dijk),
      _shotNodeIds(&shotNodes), _recNodeIds(&recNodes){
    }
    virtual ~CreateDijkstraDistMT(){}

    virtual void calc(){
        for (Index shot = start_; shot < end_; shot ++) {
            _dijkstra.setStartNode((*_shotNodeIds)[shot]);

            for (Index i = 0; i < _recNodeIds->size(); i ++) {
                (*_dists)[shot][i] = _dijkstra.distance((*_recNodeIds)[i]);
            }
        }
    }

protected:
    RMatrix             * _dists;
    Dijkstra           _dijkstra;
    const IndexArray  * _shotNodeIds;
    const IndexArray        * _recNodeIds;
};


RVector TravelTimeDijkstraModelling::response(const RVector & slowness) {
// TIC__
    if (background_ < TOLERANCE) {
        std::cout << "Background: " << background_ << "->" << 1e16 << std::endl;
        background_ = 1e16;
    }

    RVector slowPerCell(this->createMappedModel(slowness, background_));

    // J(m)*m ill only work if mesh contains a valid parametrization
    // this->createJacobian(slowPerCell);
    // return jacobian_->mult(slowness);

    dijkstra_.setGraph(createGraph(slowPerCell));

    Index nShots = shotNodeId_.size();
    Index nRecei = receNodeId_.size();
    RMatrix dMap(nShots, nRecei);

    Index nThreads = this->threadCount();

    distributeCalc(CreateDijkstraDistMT(dMap, this->dijkstra_,
                                       this->shotNodeId_,
                                       this->receNodeId_, this->verbose()),
                   nShots, nThreads, this->verbose());

    // for (Index shot = 0; shot < nShots; shot ++) {
    //     dijkstra_.setStartNode(shotNodeId_[shot]);
    //     // __MS(toc__)
    //     for (Index i = 0; i < nRecei; i ++) {
    //         dMap[shot][i] = dijkstra_.distance(receNodeId_[i]);
    //     }
    //     // exit(0);
    // }
    Index s = 0, g = 0;

    Index nData = dataContainer_->size();
    RVector resp(nData);

    for (Index dataIdx = 0; dataIdx < nData; dataIdx ++) {
        s = shotsInv_.at(Index((*dataContainer_)("s")[dataIdx]));
        g = receiInv_.at(Index((*dataContainer_)("g")[dataIdx]));
//         if (dataIdx < 10 ) std::cout << s << " " << (*dataContainer_)("s")[dataIdx] << " "
//                    << g << " " << (*dataContainer_)("g")[dataIdx] << " " << dMap[s][g] << std::endl;
        resp[dataIdx] = dMap[s][g];
    }
    return  resp;
}

void TravelTimeDijkstraModelling::initJacobian(){
    if (jacobian_ && ownJacobian_){
        delete jacobian_;
    }
    jacobian_ = new RSparseMapMatrix();
    ownJacobian_ = true;
}

const IndexArray & TravelTimeDijkstraModelling::way(Index sht, Index rec) const {
    // __MS(sht)
    // __MS(rec)
    // for (auto &x:shotsInv_ ){
    //     __MS(x.first << " " << x.second)
    // }
    // for (auto &x:receiInv_ ){
    //     __MS(x.first << " " << x.second)
    // }
    Index s = shotsInv_.at(sht);
    Index r = receiInv_.at(rec);
    // __MS(s)
    // __MS(r)
    ASSERT_SIZE(wayMatrix_, s)
    ASSERT_SIZE(wayMatrix_[s], r)
    return wayMatrix_[s][r];
}

class CreateDijkstraRowMT : public GIMLI::BaseCalcMT{
public:
    CreateDijkstraRowMT(std::vector < std::vector < IndexArray > > & wayM,
                        const Dijkstra        & dijk,
                        const IndexArray      & shotNodes,
                        const IndexArray      & recNodes,
                        bool verbose)
    : BaseCalcMT(verbose), _wayMatrix(&wayM[0]), _dijkstra(dijk),
      _shotNodeIds(&shotNodes), _recNodeIds(&recNodes){

    }

    virtual ~CreateDijkstraRowMT(){}

    virtual void calc(){
        for (Index shot = start_; shot < end_; shot ++) {
            _dijkstra.setStartNode((*_shotNodeIds)[shot]);

            for (Index i = 0; i < _recNodeIds->size(); i ++) {
                _wayMatrix[shot][i] = _dijkstra.shortestPathTo((*_recNodeIds)[i]);
            }
        }
    }

protected:
    std::vector < IndexArray > * _wayMatrix;
    Dijkstra                     _dijkstra;
    const IndexArray        * _shotNodeIds;
    const IndexArray        * _recNodeIds;
};


void TravelTimeDijkstraModelling::createJacobian(const RVector & slowness) {
    this->createJacobian(*dynamic_cast < RSparseMapMatrix * > (this->jacobian_),
                         slowness);
}

void fillWayMatrix(std::vector < std::vector < IndexArray > > & wayM,
                   const Dijkstra        & dijk,
                   const IndexArray      & shotNodes,
                   const IndexArray      & recNodes){

    ASSERT_EQUAL_SIZE(wayM, shotNodes)
    ASSERT_EQUAL_SIZE(wayM[0], recNodes)

    Dijkstra _dijkstra(dijk);

    for (Index shot = 0; shot < shotNodes.size(); shot ++) {
        _dijkstra.setStartNode(shotNodes[shot]);
#pragma omp parallel if (useOMP())
{

    //print("A:", omp_get_thread_num());
        for (Index i = 0; i < recNodes.size(); i ++) {
            wayM[shot][i] = _dijkstra.shortestPathTo((recNodes)[i]);
        }
}
    }
}

void TravelTimeDijkstraModelling::createJacobian(RSparseMapMatrix & jacobian,
                                                 const RVector & slowness) {

    if (min(this->mesh_->cellMarkers()) < 0){
        log(Warning, "There are cells with marker -1. "
                "Did you define a boundary region? (This is not needed).");
    }
    Stopwatch swatch(true);
    if (background_ < TOLERANCE) {
        std::cout << "Background: " << background_ << " ->" << 1e16 << std::endl;
        background_ = 1e16;
    }
    RVector slowPerCell(this->createMappedModel(slowness, background_));

    dijkstra_.setGraph(createGraph(slowPerCell));

    Index nShots = shotNodeId_.size();
    Index nRecei = receNodeId_.size();
    Index nData = dataContainer_->size();
    Index nModel = slowness.size();

    jacobian.clear();
    jacobian.setRows(nData);
    jacobian.setCols(nModel);

    //** for each shot: vector<  way(shot->geoph) >;
    wayMatrix_.clear();
    wayMatrix_.resize(nShots);
    for (auto & w: wayMatrix_) w.resize(nRecei);

    Index nThreads = this->threadCount();

    if (useOMP()){
        __MS("DEBUG: OMP for fill Way Matrix")
        fillWayMatrix(wayMatrix_, dijkstra_, shotNodeId_, receNodeId_);
    } else {
        distributeCalc(CreateDijkstraRowMT(wayMatrix_, dijkstra_,
                                       shotNodeId_, receNodeId_, this->verbose()),
                    nShots, nThreads, this->verbose());
    }

    if (this->verbose()){
        std::cout << "/" << swatch.duration(true);
    }
    // for (Index shot = 0; shot < nShots; shot ++) {
    //     dijkstra_.setStartNode(shotNodeId_[shot]);

    //     for (Index i = 0; i < nRecei; i ++) {
    //         wayMatrix_[shot].push_back(dijkstra_.shortestPathTo(receNodeId_[i]));
    //     }
    // }
    for (Index dataIdx = 0; dataIdx < nData; dataIdx ++) {
        Index s = shotsInv_.at(Index((*dataContainer_)("s")[dataIdx]));
        Index g = receiInv_.at(Index((*dataContainer_)("g")[dataIdx]));

        std::set < Cell * > neighborCells;

        for (Index i = 0; i < wayMatrix_[s][g].size()-1; i ++) {
            neighborCells.clear();

            Index aId = wayMatrix_[s][g][i];
            Index bId = wayMatrix_[s][g][i + 1];

            const GraphDistInfo & way = dijkstra_.graphInfo(aId, bId);

            double edgeLength = way.dist();
            //double edgeLength = mesh_->node(aId).pos().distance(mesh_->node(bId).pos());
            // double slo = 0.0;

            double minSlow = 9e99;

            for (const auto &iCD : way.cellIDs()){
                minSlow = min(minSlow, slowPerCell[iCD]);
            }

            for (const auto &iCD : way.cellIDs()){
                if (std::fabs(slowPerCell[iCD] - minSlow) < 1e-4){
                    Cell *c = & mesh_->cell(iCD);
                    neighborCells.insert(c);
                }
            }

            for (const auto &c : neighborCells){
                jacobian[dataIdx][c->marker()] += edgeLength / neighborCells.size();
            }
        }
    }
    if (this->verbose()){
        std::cout << "/" << swatch.duration(true) << " ";
        std::cout << std::endl;
    }
}

TTModellingWithOffset::TTModellingWithOffset(Mesh & mesh, DataContainer & dataContainer, bool verbose)
: TravelTimeDijkstraModelling(mesh, dataContainer, verbose) {

    //! find occuring shots, and map them to indices starting from zero
    shots_ = unique(sort(dataContainer.get("s")));
    std::cout << "found " << shots_.size() << " shots." << std::endl;
    for (Index i = 0 ; i < shots_.size() ; i++) {
        shotMap_.insert(std::pair< Index, Index >((Index)shots_[i], i));
    }

    //! create new region containing offsets with special marker

    offsetMesh_ = createMesh1D(shots_.size());
    for (size_t i = 0 ; i < offsetMesh_.cellCount() ; i++) {
        offsetMesh_.cell(i).setMarker(NEWREGION);
    }

    regionManager().addRegion(NEWREGION, offsetMesh_, NEWREGION);

    this->initJacobian();
}

TTModellingWithOffset::~TTModellingWithOffset() { }

RVector TTModellingWithOffset::createDefaultStartModel() {
    return cat(TravelTimeDijkstraModelling::createDefaultStartModel(), RVector(shots_.size()));
}

RVector TTModellingWithOffset::response(const RVector & model) {
    //! extract slowness from model and call old function

    RVector slowness(model, 0, model.size() - shots_.size());
    RVector offsets(model, model.size() - shots_.size(), model.size());
    RVector resp = TravelTimeDijkstraModelling::response(slowness); //! normal response
    RVector shotpos = dataContainer_->get("s");

    for (size_t i = 0; i < resp.size() ; i++){
        resp[i] += offsets[shotMap_[Index(shotpos[i])]];
    }

    return resp;
}

void TTModellingWithOffset::initJacobian(){
    if (jacobian_ && ownJacobian_){
        delete jacobian_;
    }
    jacobian_ = new H2SparseMapMatrix();
    ownJacobian_ = true;
}

void TTModellingWithOffset::createJacobian(const RVector & model){

    H2SparseMapMatrix *jacobian = dynamic_cast < H2SparseMapMatrix* > (jacobian_);
    //! extract slowness from model and call old function

    RVector slowness(model, 0, model.size() - shots_.size());
    RVector offsets(model, model.size() - shots_.size(), model.size());

    TravelTimeDijkstraModelling::createJacobian(jacobian->H1(), slowness);
    jacobian->H2().setRows(dataContainer_->size());
    jacobian->H2().setCols(offsets.size());

    //! set 1 entries for the used shot
    RVector shotpos = dataContainer_->get("s"); // shot=C1/A

    for (size_t i = 0; i < dataContainer_->size(); i++) {
        jacobian->H2().setVal(i, shotMap_[Index(shotpos[i])], 1.0);
    }
}

TravelTimeDijkstraModellingTTI::TravelTimeDijkstraModellingTTI(bool verbose)
    : TravelTimeDijkstraModelling(verbose){
    this->dEpsLookup = 0.01;
    this->dDeltaLookup = 0.01;
    // 1°
    this->dAngleLookup = M_PI / 180.;
    this->stepsAngleLookup = std::round(0.5 * M_PI / this->dAngleLookup);
    this->isLookupComputed = false;
}

TravelTimeDijkstraModellingTTI::TravelTimeDijkstraModellingTTI(Mesh & mesh, DataContainer & dataContainer, bool verbose) :
    TravelTimeDijkstraModelling(mesh, dataContainer, verbose) {
    this->dEpsLookup = 0.01;
    this->dDeltaLookup = 0.01;
    // 1°
    this->dAngleLookup = M_PI / 180.;
    this->stepsAngleLookup = std::round(0.5 * M_PI / this->dAngleLookup);
    this->isLookupComputed = false;
}

TravelTimeDijkstraModellingTTI::~TravelTimeDijkstraModellingTTI() { }

RVector TravelTimeDijkstraModellingTTI::response(const RVector & combined_model) {
    RVector velP0, epsilon, delta, incl, azim;
    assert(combined_model.size() % 5 == 0);
    int paramSize = combined_model.size() / 5;

    velP0.resize(paramSize);
    epsilon.resize(paramSize);
    delta.resize(paramSize);
    incl.resize(paramSize);
    azim.resize(paramSize);

    // copy data (only works without bound check in subscription operator see Vector::copy_)
    std::copy(&combined_model[0], &combined_model[paramSize], &velP0[0]);
    std::copy(&combined_model[paramSize], &combined_model[2 * paramSize], &epsilon[0]);
    std::copy(&combined_model[2 * paramSize], &combined_model[3 * paramSize], &delta[0]);
    std::copy(&combined_model[3 * paramSize], &combined_model[4 * paramSize], &incl[0]);
    std::copy(&combined_model[4 * paramSize], &combined_model[5 * paramSize], &azim[0]);

    // call actual function
    return this->response(velP0, epsilon, delta, incl, azim);
}


RVector TravelTimeDijkstraModellingTTI::response(const RVector & vel0, const RVector & epsilon,
    const RVector & delta, const RVector & incl, const RVector & azim) {

    if (background_ < TOLERANCE) {
        std::cout << "Background: " << background_ << "->" << 1e16 << std::endl;
        background_ = 1e16;
    }

    // make sure the lookup table is set up
    this->testRecomputeLookup(epsilon, delta);

    RVector velPerCell(this->createMappedModel(vel0, background_));
    RVector epsPerCell(this->createMappedModel(epsilon, background_));
    RVector delPerCell(this->createMappedModel(delta, background_));
    RVector inclPerCell(this->createMappedModel(incl, background_));
    RVector azimPerCell(this->createMappedModel(azim, background_));

    dijkstra_.setGraph(createGraph(velPerCell, epsPerCell, delPerCell, inclPerCell, azimPerCell));

    Index nShots = shotNodeId_.size();
    Index nRecei = receNodeId_.size();
    RMatrix dMap(nShots, nRecei);

    Index nThreads = this->threadCount();

    distributeCalc(CreateDijkstraDistMT(dMap, this->dijkstra_,
                                       this->shotNodeId_,
                                       this->receNodeId_, this->verbose()),
                   nShots, nThreads, this->verbose());

    Index s = 0, g = 0;

    Index nData = dataContainer_->size();
    RVector resp(nData);

    for (Index dataIdx = 0; dataIdx < nData; dataIdx ++) {
        s = shotsInv_.at(Index((*dataContainer_)("s")[dataIdx]));
        g = receiInv_.at(Index((*dataContainer_)("g")[dataIdx]));
        resp[dataIdx] = dMap[s][g];
    }
    return  resp;
}

void TravelTimeDijkstraModellingTTI::createJacobian(const RVector & combined_model) {
    RVector velP0, epsilon, delta, incl, azim;
    assert(combined_model.size() % 5 == 0);
    int paramSize = combined_model.size() / 5;

    velP0.resize(paramSize);
    epsilon.resize(paramSize);
    delta.resize(paramSize);
    incl.resize(paramSize);
    azim.resize(paramSize);

    // copy data (only works without bound check in subscription operator see Vector::copy_)
    std::copy(&combined_model[0], &combined_model[paramSize], &velP0[0]);
    std::copy(&combined_model[paramSize], &combined_model[2 * paramSize], &epsilon[0]);
    std::copy(&combined_model[2 * paramSize], &combined_model[3 * paramSize], &delta[0]);
    std::copy(&combined_model[3 * paramSize], &combined_model[4 * paramSize], &incl[0]);
    std::copy(&combined_model[4 * paramSize], &combined_model[5 * paramSize], &azim[0]);

    // call actual function
    this->createJacobian(velP0, epsilon, delta, incl, azim);
}

void TravelTimeDijkstraModellingTTI::createJacobian(const RVector & velP0, const RVector & epsilon,
    const RVector & delta, const RVector & incl, const RVector & azim) {
    this->createJacobian(*dynamic_cast < RSparseMapMatrix * > (this->jacobian_),
                         velP0, epsilon, delta, incl, azim);
}

double TravelTimeDijkstraModellingTTI::ttiToSlowness3D(double V0, double epsilon, double delta, double incl, double azim,
        double pathX, double pathY, double pathZ) {

    double vecPathLen = std::sqrt(pathX * pathX + pathY * pathY + pathZ * pathZ);

    double phi;
    // Avoid division by zero if symmLen or vecPathLen is zero
    if (vecPathLen < TOLERANCE) {
        phi = M_PI / 2.0; // Assume perpendicular if one vector is zero
    } else {
        double cosIncl = std::cos(incl);
        double cosAzim = std::cos(azim);
        double sinIncl = std::sin(incl);
        double sinAzim = std::sin(azim);

        phi = std::acos(-(pathX * sinIncl * cosAzim + pathY * sinIncl * sinAzim + pathZ * cosIncl) / vecPathLen);
    }

    return 1. / (this->interpolateAnisotropyScalar(epsilon, delta, phi) * V0);
}
double TravelTimeDijkstraModellingTTI::ttiToSlowness2D(double V0, double epsilon, double delta, double incl, double pathX, double pathY) {

    double vecPathLen = std::sqrt(pathX * pathX + pathY * pathY);

    double phi;
    // Avoid division by zero if symmLen or vecPathLen is zero
    if (vecPathLen < TOLERANCE) {
        phi = M_PI * 0.5; // Assume perpendicular if one vector is zero
    } else {
        double cosIncl = std::cos(incl);
        double sinIncl = std::sin(incl);

        phi = std::acos(-(pathX * sinIncl + pathY * cosIncl) / vecPathLen);
    }
                
    return 1. / (this->interpolateAnisotropyScalar(epsilon, delta, phi) * V0);
}

void ttiJacobianEntry3D(double V0, double epsilon, double delta, double incl, double azim, double pathX, double pathY, double pathZ,
    double& dTdV0, double& dTdEpsilon, double& dTdDelta, double& dTdIncl, double& dTdAzim) {

    double sinIncl = std::sin(incl);
    double sinAzim = std::sin(azim);
    double cosIncl = std::cos(incl);
    double cosAzim = std::cos(azim);
    double vecPathLen = std::sqrt(pathX * pathX + pathY * pathY + pathZ * pathZ);
    double cosTheta = (pathX * sinIncl * cosAzim + pathY * sinIncl * sinAzim - pathZ * cosIncl) / vecPathLen;
                
    // Avoid division by zero if vecPathLen is zero
    double theta;
    if (vecPathLen < TOLERANCE) {
        theta = M_PI / 2.0; // Assume perpendicular if one vector is zero
    } else {
        theta = std::acos(cosTheta);
    }

    double sinTheta = std::sin(theta);
    double sinThetaSq = sinTheta * sinTheta;
    double cosThetaSq = cosTheta * cosTheta;

    double substA = 0.5 + epsilon * sinThetaSq;
    double substB = 2. * sinThetaSq * cosThetaSq;
    double substC = std::sqrt((substA * substA) + substB * (delta - epsilon));

    double lambdaAc = std::sqrt(substA + substC);

    double VP = V0 * lambdaAc;

    // partial derivatives of velocity model parameters
    //// V0
    double dTdS = vecPathLen;
    double dTdVP = -dTdS / (VP * VP);
    dTdV0 = dTdVP * lambdaAc;

    //// Epsilon
    double dTdL = dTdVP * V0;
    double dLda_c = 1. / (2. * lambdaAc);
    double dadE = sinThetaSq;
    double dcdE = (2. * substA * sinThetaSq - substB) / (2. * substC);
    double dLdE = (dLda_c * dadE) + (dLda_c * dcdE);
    dTdEpsilon = dTdL * dLdE;

    //// Delta
    double dTdc = dTdL * dLda_c;
    dTdDelta = dTdc * (substB / (2. * substC));

    //// incl and azim
    double dcda = substA / substC;
    double dcdb = (delta - epsilon) / (2. * substC);
    double dadTheta = 2. * epsilon * sinTheta * cosTheta;
    double dbdTheta = 4. * ((sinTheta * cosThetaSq * cosTheta) - (sinThetaSq * sinTheta * cosTheta));
    double dThetadk = -1. / std::sqrt(1. - cosThetaSq);
    double dkdIncl = (pathX * cosIncl * cosAzim  + pathY * cosIncl * sinAzim + pathZ * sinIncl) / vecPathLen;
    double dkdAzim = (-pathX * sinIncl * sinAzim + pathY * sinIncl * cosAzim) / vecPathLen;
    double dadk = dadTheta * dThetadk;
    double dbdk = dbdTheta * dThetadk;
    dTdIncl = dTdL * (dLda_c * dadk * dkdIncl + dLda_c * (dcda * dadk * dkdIncl + dcdb * dbdk * dkdIncl));
    dTdAzim = dTdL * (dLda_c * dadk * dkdAzim + dLda_c * (dcda * dadk * dkdAzim + dcdb * dbdk * dkdAzim));
}

void ttiJacobianEntry2D(double V0, double epsilon, double delta, double incl, double pathX, double pathY,
    double& dTdV0, double& dTdEpsilon, double& dTdDelta, double& dTdIncl) {

    double sinIncl = std::sin(incl);
    double cosIncl = std::cos(incl);
    double vecPathLen = std::sqrt(pathX * pathX + pathY * pathY);
    double cosTheta = (pathX * sinIncl - pathY * cosIncl) / vecPathLen;
                
    // Avoid division by zero if vecPathLen is zero
    double theta;
    if (vecPathLen < TOLERANCE) {
        theta = M_PI / 2.0; // Assume perpendicular if one vector is zero
    } else {
        theta = std::acos(cosTheta);
    }

    double sinTheta = std::sin(theta);
    double sinThetaSq = sinTheta * sinTheta;
    double cosThetaSq = cosTheta * cosTheta;

    double substA = 0.5 + epsilon * sinThetaSq;
    double substB = 2. * sinThetaSq * cosThetaSq;
    double substC = std::sqrt((substA * substA) + substB * (delta - epsilon));

    double lambdaAc = std::sqrt(substA + substC);

    double VP = V0 * lambdaAc;

    // partial derivatives of velocity model parameters
    //// V0
    double dTdS = vecPathLen;
    double dTdVP = -dTdS / (VP * VP);
    dTdV0 = dTdVP * lambdaAc;

    //// Epsilon
    double dTdL = dTdVP * V0;
    double dLda_c = 1. / (2. * lambdaAc);
    double dadE = sinThetaSq;
    double dcdE = (2. * substA * sinThetaSq - substB) / (2. * substC);
    double dLdE = (dLda_c * dadE) + (dLda_c * dcdE);
    dTdEpsilon = dTdL * dLdE;

    //// Delta
    double dTdc = dTdL * dLda_c;
    dTdDelta = dTdc * (substB / (2. * substC));

    //// incl
    double dcda = substA / substC;
    double dcdb = (delta - epsilon) / (2. * substC);
    double dadTheta = 2. * epsilon * sinTheta * cosTheta;
    double dbdTheta = 4. * (sinTheta * cosThetaSq * cosTheta - sinThetaSq * sinTheta * cosTheta);
    double dThetadk = -1. / std::sqrt(1. - cosThetaSq);
    double dkdIncl = (pathX * cosIncl + pathY * sinIncl) / vecPathLen;
    double dadk = dadTheta * dThetadk;
    double dbdk = dbdTheta * dThetadk;
    dTdIncl = dTdL * (dLda_c * dadk * dkdIncl + dLda_c * (dcda * dadk * dkdIncl + dcdb * dbdk * dkdIncl));
}

void TravelTimeDijkstraModellingTTI::createJacobian(RSparseMapMatrix & jacobian, const RVector & velP0, const RVector & epsilon,
    const RVector & delta, const RVector & incl, const RVector & azim) {

    if (min(this->mesh_->cellMarkers()) < 0){
        log(Warning, "There are cells with marker -1. "
                "Did you define a boundary region? (This is not needed).");
    }
    Stopwatch swatch(true);
    if (background_ < TOLERANCE) {
        std::cout << "Background: " << background_ << " ->" << 1e16 << std::endl;
        background_ = 1e16;
    }

    double sumVp0 = 0.;
    double sumEps = 0.;
    double sumDel = 0.;
    double sumIncl = 0.;
    double sumAzim = 0.;

    RVector velPerCell(this->createMappedModel(velP0, background_));
    RVector epsPerCell(this->createMappedModel(epsilon, background_));
    RVector delPerCell(this->createMappedModel(delta, background_));
    RVector inclPerCell(this->createMappedModel(incl, background_));
    RVector azimPerCell(this->createMappedModel(azim, background_));

    dijkstra_.setGraph(createGraph(velPerCell, epsPerCell, delPerCell, inclPerCell, azimPerCell));

    Index nShots = shotNodeId_.size();
    Index nRecei = receNodeId_.size();
    Index nData = dataContainer_->size();
    Index nModel = velP0.size();

    jacobian.clear();
    jacobian.setRows(nData);
    jacobian.setCols(nModel * 5);

    //** for each shot: vector<  way(shot->geoph) >;
    wayMatrix_.clear();
    wayMatrix_.resize(nShots);
    for (auto & w: wayMatrix_) w.resize(nRecei);

    Index nThreads = this->threadCount();

    if (useOMP()){
        __MS("DEBUG: OMP for fill Way Matrix")
        fillWayMatrix(wayMatrix_, dijkstra_, shotNodeId_, receNodeId_);
    } else {
        distributeCalc(CreateDijkstraRowMT(wayMatrix_, dijkstra_,
                                       shotNodeId_, receNodeId_, this->verbose()),
                    nShots, nThreads, this->verbose());
    }

    if (this->verbose()){
        std::cout << "/" << swatch.duration(true);
    }
    
    for (Index dataIdx = 0; dataIdx < nData; dataIdx ++) {
        Index s = shotsInv_.at(Index((*dataContainer_)("s")[dataIdx]));
        Index g = receiInv_.at(Index((*dataContainer_)("g")[dataIdx]));

        std::vector < Cell * > neighborCells;
        std::vector < Index > neighborCellIDs;
        std::vector < double > wayCellSlowness;
        std::vector < Index > wayCellIDs;

        for (Index i = 0; i < wayMatrix_[s][g].size()-1; i ++) {
            neighborCells.clear();
            neighborCellIDs.clear();
            wayCellSlowness.clear();
            wayCellIDs.clear();

            Index aId = wayMatrix_[s][g][i];
            Index bId = wayMatrix_[s][g][i + 1];

            // node positions
            RVector3 nodeA_pos = mesh_->node(aId).pos();
            RVector3 nodeB_pos = mesh_->node(bId).pos();

            // compute vector between nodes
            RVector3 vecPath = nodeB_pos - nodeA_pos;

            const GraphDistInfo & way = dijkstra_.graphInfo(aId, bId);

            double edgeLength = way.dist();

            double minSlow = 9e99;

            auto cellIDs = way.cellIDs();
            for (const auto &iCD : cellIDs){
                double cellSlowness;
                if (mesh_->dim() == 3) {
                    cellSlowness = this->ttiToSlowness3D(
                        velPerCell[iCD],
                        epsPerCell[iCD],
                        delPerCell[iCD],
                        inclPerCell[iCD], 
                        azimPerCell[iCD], 
                        vecPath.x(),
                        vecPath.y(),
                        vecPath.z());
                }
                else {
                    cellSlowness = this->ttiToSlowness2D(
                        velPerCell[iCD],
                        epsPerCell[iCD],
                        delPerCell[iCD],
                        inclPerCell[iCD], 
                        vecPath.x(),
                        vecPath.y());

                }
                wayCellSlowness.push_back(cellSlowness);
                wayCellIDs.push_back(iCD);

                minSlow = min(minSlow, cellSlowness);
            }

            for (std::size_t i = 0; i < cellIDs.size(); i++){
                auto iCD = wayCellIDs[i];
                double cellSlowness = wayCellSlowness[i];
                if (std::fabs(cellSlowness- minSlow) < 1e-4){
                    Cell *c = & mesh_->cell(iCD);
                    neighborCells.push_back(c);
                    neighborCellIDs.push_back(iCD);
                }
            }

            double scaleEntry = 1. / neighborCells.size();

            for (std::size_t i = 0; i < neighborCells.size(); i++) {
                const auto c = neighborCells[i];
                const auto cellID = neighborCellIDs[i];

                double dTdV0, dTdEpsilon, dTdDelta, dTdIncl, dTdAzim;
                
                if (mesh_->dim() == 3) {
                    ttiJacobianEntry3D(
                        velPerCell[cellID],
                        epsPerCell[cellID],
                        delPerCell[cellID],
                        inclPerCell[cellID],
                        azimPerCell[cellID],
                        vecPath.x(),
                        vecPath.y(),
                        vecPath.z(),
                        dTdV0,
                        dTdEpsilon,
                        dTdDelta,
                        dTdIncl,
                        dTdAzim);
                }
                else {
                    ttiJacobianEntry2D(
                        velPerCell[cellID],
                        epsPerCell[cellID],
                        delPerCell[cellID],
                        inclPerCell[cellID],
                        vecPath.x(),
                        vecPath.y(),
                        dTdV0,
                        dTdEpsilon,
                        dTdDelta,
                        dTdIncl);
                    dTdAzim = 0.0;
                }

                // entry for P-wave velocity
                jacobian[dataIdx][c->marker()] += dTdV0 * scaleEntry;

                // entry for epsilon
                jacobian[dataIdx][c->marker() + nModel] += dTdEpsilon * scaleEntry;

                // entry for delta
                jacobian[dataIdx][c->marker() + 2 * nModel] += dTdDelta * scaleEntry;

                // entry for inclination of symmetry axis
                jacobian[dataIdx][c->marker() + 3 * nModel] += dTdIncl * scaleEntry;

                // entry for azimuth of symmetry axis
                jacobian[dataIdx][c->marker() + 4 * nModel] += dTdAzim * scaleEntry;

                sumVp0 += std::abs(dTdV0);
                sumEps += std::abs(dTdEpsilon);
                sumDel += std::abs(dTdDelta);
                sumIncl += std::abs(dTdIncl);
                sumAzim += std::abs(dTdAzim);
            }
        }
    }
    std::cout << std::endl << "Sum abs vp0: " << sumVp0 << std::endl;
    std::cout << "Sum abs eps: " << sumEps << std::endl;
    std::cout << "Sum abs del: " << sumDel << std::endl;
    std::cout << "Sum abs incl: " << sumIncl << std::endl;
    std::cout << "Sum abs azim: " << sumAzim << std::endl;

    if (this->verbose()){
        std::cout << "/" << swatch.duration(true) << " ";
        std::cout << std::endl;
    }
}

void TravelTimeDijkstraModellingTTI::fillGraph_(Graph & graph, const Node & a, const Node & b, double slowness, SIndex leftID){
    if (a.id() == b.id()) return;

    double dist = a.pos().distance(b.pos());

    // ensure connection between 3d boundaries
    dist = max(1e-8, dist);

    double newTime = dist * slowness;
    double oldTime = graph[a.id()][b.id()].time();

    // if (V_){
    //     __MS("a:" << a.id() << " b:"  << b.id() << " L:" << leftID << " t:" << " " << newTime << " " << oldTime)
    // }

    if (oldTime > 0.0) {
        newTime = std::min(newTime, oldTime);

        // way pair already exist so set time to min and add leftID

        NodeDistMap::iterator ita(graph[a.id()].find(b.id()));
        ita->second.cellIDs().insert(leftID);
        ita->second.setTime(newTime);

        NodeDistMap::iterator itb(graph[b.id()].find(a.id()));
        itb->second.cellIDs().insert(leftID);
        itb->second.setTime(newTime);

    } else {
        // first time fill
        graph[a.id()][b.id()] = GraphDistInfo(newTime, dist, leftID);
        graph[b.id()][a.id()] = GraphDistInfo(newTime, dist, leftID);
    }
}

void TravelTimeDijkstraModellingTTI::fillGraph_(Graph & graph, Cell & c, double slowness){

    std::vector< Node * > ni(c.nodes());

    for (Index i(0); i < c.boundaryCount(); i++){
        Boundary *b = c.boundary(i);
        if (b){
            for (auto & n : b->secondaryNodes()){
                ni.push_back(n);
            }
        } else {
            log(Critical, "No boundary found.");
        }
    }

    for (auto & n : c.secondaryNodes()){
        ni.push_back(n);
    }

    for (Index j = 0; j < ni.size()-1; j ++) {
        for (Index k = j + 1; k < ni.size(); k ++) {
            this->fillGraph_(graph, *ni[j], *ni[k], slowness, c.id());
        }
    }
}

void TravelTimeDijkstraModellingTTI::fillGraph_(Graph & graph, Cell & c, double vel0, double epsilon,
        double delta, double incl, double azim, bool is3D){

    double slowness;
    RVector3 vecPath; // vector between two nodes

    std::vector< Node * > ni(c.nodes());

    for (Index i(0); i < c.boundaryCount(); i++){
        Boundary *b = c.boundary(i);
        if (b){
            for (auto & n : b->secondaryNodes()){
                ni.push_back(n);
            }
        } else {
            log(Critical, "No boundary found.");
        }
    }

    for (auto & n : c.secondaryNodes()){
        ni.push_back(n);
    }

    for (Index j = 0; j < ni.size()-1; j ++) {
        for (Index k = j + 1; k < ni.size(); k ++) {
            // compute wavefront path
            vecPath = ni[k]->pos() - ni[j]->pos();
            // compute slowness
            if (is3D) {
                slowness = this->ttiToSlowness3D(vel0, epsilon, delta, incl, azim, vecPath.x(), vecPath.y(), vecPath.z());
            }
            else {
                slowness = this->ttiToSlowness2D(vel0, epsilon, delta, incl, vecPath.x(), vecPath.y());
            }
            
            /*
            std::string log_str = "Path:\n\tA  : [" + std::to_string(ni[j]->pos().x()) + ", " + std::to_string(ni[j]->pos().y()) + ", " + std::to_string(ni[j]->pos().z()) + "]\n\t" +
                         "B  : [" + std::to_string(ni[k]->pos().x()) + ", " + std::to_string(ni[k]->pos().y()) + ", " + std::to_string(ni[k]->pos().z()) + "]\n\t" +
                         "B-A: [" + std::to_string(vecPath.x()) + ", " + std::to_string(vecPath.y()) + ", " + std::to_string(vecPath.z()) + "]\n\t" +
                         "VP0: " + std::to_string(vel0) + "\n\t" +
                         "eps: " + std::to_string(epsilon) + "\n\t" +
                         "del: " + std::to_string(delta) + "\n\t" +
                         "incl: " + std::to_string(incl) + "\n\t" +
                         "azim: " + std::to_string(azim) + "\n\t" +
                         "velocity: " + std::to_string(1 / slowness) + "\n";
            std::cout << log_str << std::endl;
            if (k > 10) {
                std::exit(0);
            }
            */

            this->fillGraph_(graph, *ni[j], *ni[k], slowness, c.id());
        }
    }
}

Graph TravelTimeDijkstraModellingTTI::createGraph(const RVector & velPerCell, const RVector & epsPerCell,
    const RVector & delPerCell, const RVector & inclPerCell, const RVector & azimPerCell) {

    Graph graph;
    mesh_->createNeighborInfos();

    for (Index i = 0; i < mesh_->cellCount(); i ++) {
        Cell & c = mesh_->cell(i);
        this->fillGraph_(graph, c, velPerCell[c.id()], epsPerCell[c.id()], delPerCell[c.id()], inclPerCell[c.id()], azimPerCell[c.id()], mesh_->dim() == 3);
    }

    if (graph.size() < mesh_->nodeCount()){
        std::cerr << WHERE_AM_I <<
                " there seems to be unassigned nodes within the mesh. Dijkstra Path will be maybe invalid."
                 << graph.size() << " < " << mesh_->nodeCount() << std::endl;
    }
    return graph;
}

// this could be optimized by not recomputing the whole table but only adding the new entries, but it would involve some nasty moving operations, which I don't want to do until it is really necessary.
void TravelTimeDijkstraModellingTTI::testRecomputeLookup(const RVector & epsilon, const RVector & delta) {
    double minEps = 999.;
    double maxEps = -999.;
    double minDelta = 999.;
    double maxDelta = -999.;
    
    for(Index i = 0; i < epsilon.size(); i++) {
        minEps = minEps > epsilon[i] ? epsilon[i] : minEps;
        maxEps = maxEps < epsilon[i] ? epsilon[i] : maxEps;
    }
    for(Index i = 0; i < delta.size(); i++) {
        minDelta = minDelta > delta[i] ? delta[i] : minDelta;
        maxDelta = maxDelta < delta[i] ? delta[i] : maxDelta;
    }

    // check if min/max of epsilon and delta lie in range of current lookup table and exit if that is the case
    if(this->isLookupComputed && minEps >= this->minEpsLookup && maxEps <= this->maxEpsLookup &&
        minDelta >= this->minDeltaLookup && maxDelta <= this->maxDeltaLookup) {
            return;
        }

    // else: (re)compute larger lookup table to contain anisotropy scalars for all cases that are present in the data

    // add a little bit of buffer
    this->minEpsLookup = minEps - 20 * this->dEpsLookup;
    this->maxEpsLookup = maxEps + 20 * this->dEpsLookup;
    this->minDeltaLookup = minDelta - 20 * this->dDeltaLookup;
    this->maxDeltaLookup = maxDelta + 20 * this->dDeltaLookup;

    this->stepsEpsLookup = std::floor((this->maxEpsLookup - this->minEpsLookup) / this->dEpsLookup) + 1;
    this->stepsDeltaLookup = std::floor((this->maxDeltaLookup - this->minDeltaLookup) / this->dDeltaLookup) + 1;

    // adapt maximum value
    this->maxEpsLookup = this->minEpsLookup + this->dEpsLookup * this->stepsEpsLookup;
    this->maxDeltaLookup = this->minDeltaLookup + this->dDeltaLookup * this->stepsDeltaLookup;

    std::cout << "(Re)computing lookup table for epsilon in [" << this->minEpsLookup << ", " << this->maxEpsLookup << "] and delta in [" << this->minDeltaLookup << ", " << this->maxDeltaLookup << "]...";

    // resize lookup table
    this->anisotropyScalarLookup.resize(this->stepsEpsLookup * this->stepsDeltaLookup * this->stepsAngleLookup);

    // initialize the global index of the lookup table
    Index idxLookup = 0;

    // helper variables for the inner loops
    double epsCurrent, deltaCurrent, phiCurrent, thetaCurrent, sinTheta, cosTheta, sinThetaSq, cosThetaSq;
    double lbd_ell_sq, lbd_ac, lbd_ac_sq, lbd_big, d_lbd, gx, gz, anisotropyScalar;
    double theta_comparison;
    double phi_comparison;
    double phi_comparison_last = 0.;

    // fill the table
    for(Index idxEps = 0; idxEps < this->stepsEpsLookup; idxEps++) {
        epsCurrent =  idxEps * this->dEpsLookup + this->minEpsLookup;

        for(Index idxDelta = 0; idxDelta < this->stepsDeltaLookup; idxDelta++) { 
            deltaCurrent =  idxDelta * this->dDeltaLookup + this->minDeltaLookup;

            // loop over group angle phi
            for(Index idxPhi = 0; idxPhi < this->stepsAngleLookup; idxPhi++) {
                phiCurrent = idxPhi * this->dAngleLookup;

                // find phase angle theta that corresponds to the current group angle phi
                if(idxPhi == 0) {
                    theta_comparison = 0.;
                }
                else if(idxPhi == this->stepsAngleLookup - 1) {
                    theta_comparison = M_PI * 0.5;
                }
                //// loop over phase angles and compute corresponding group angle. Interpolate if necessary.
                else {
                    for(Index idxTheta = 1; idxTheta < this->stepsAngleLookup - 1; idxTheta++) {
                        thetaCurrent = idxTheta * this->dAngleLookup;
                        sinTheta = std::sin(thetaCurrent);
                        cosTheta = std::cos(thetaCurrent);
                        sinThetaSq = sinTheta * sinTheta;
                        cosThetaSq = cosTheta * cosTheta;

                        // compute corresponding group angle
                        lbd_ell_sq = 2. * epsCurrent * sinThetaSq + 1.;
                        lbd_ac_sq = lbd_ell_sq * 0.5 + std::sqrt(lbd_ell_sq * lbd_ell_sq * 0.25 - 2. * (epsCurrent - deltaCurrent) * sinThetaSq * cosThetaSq);
                        lbd_big = (epsCurrent + (lbd_ell_sq * epsCurrent - 2. * (epsCurrent - deltaCurrent) * std::cos(2. * thetaCurrent))) / (2. * lbd_ac_sq - lbd_ell_sq);
                        gx = sinTheta * (lbd_ac_sq + cosThetaSq * lbd_big);
                        gz = cosTheta * (lbd_ac_sq - sinThetaSq * lbd_big);
                        phi_comparison = std::atan2(gx, gz);

                        // interpolate phase angle that corresponds to group angle
                        if(phi_comparison == phiCurrent) {
                            theta_comparison = thetaCurrent;
                            break;
                        }
                        else if(phi_comparison > phiCurrent) {
                            theta_comparison = thetaCurrent - this->dAngleLookup * ((phi_comparison - phiCurrent) / (phi_comparison - phi_comparison_last));
                            break;
                        }
                    }
                }

                // compute anisotropic amplification scalar
                sinTheta = std::sin(theta_comparison);
                cosTheta = std::cos(theta_comparison);
                sinThetaSq = sinTheta * sinTheta;
                cosThetaSq = cosTheta * cosTheta;

                lbd_ell_sq = 2. * epsCurrent * sinThetaSq + 1.;
                lbd_ac = std::sqrt(lbd_ell_sq * 0.5 + std::sqrt(lbd_ell_sq * lbd_ell_sq * 0.25 - 2. * (epsCurrent - deltaCurrent) * sinThetaSq * cosThetaSq));
                lbd_ac_sq = lbd_ac * lbd_ac;
                lbd_big = (epsCurrent + (lbd_ell_sq * epsCurrent - 2. * (epsCurrent - deltaCurrent) * std::cos(2. * theta_comparison))) / (2. * lbd_ac_sq - lbd_ell_sq);
                d_lbd = 0.5 * std::sin(theta_comparison * 2.) / lbd_ac * lbd_big;

                //anisotropyScalar = 1. / std::sqrt(lbd_ac_sq + d_lbd * d_lbd);
                anisotropyScalar = std::sqrt(lbd_ac_sq + lbd_ac_sq * d_lbd * d_lbd);

                // write into lookup table
                this->anisotropyScalarLookup[idxLookup++] = anisotropyScalar;
            }
        }
    }

    std::cout << " lookup table computed!" << std::endl;
    this->isLookupComputed = true;
}

// trilinear interpolation of lookup table
double TravelTimeDijkstraModellingTTI::interpolateAnisotropyScalar(double epsilon, double delta, double groupAngle) {
    // angle might be [-pi, pi], but weak anisotropy only works for [0, pi/2]
    if(groupAngle < 0.) {
        groupAngle = -groupAngle;
    }
    if(groupAngle > M_PI * 0.5) {
        groupAngle = M_PI - groupAngle;
    }

    // boundary indices of nodes surrounding the position
    Index idxEps1 = std::floor((epsilon - this->minEpsLookup) / this->dEpsLookup);
    Index idxEps2 = idxEps1 + 1;
    Index idxDelta1 = std::floor((delta - this->minDeltaLookup) / this->dDeltaLookup);
    Index idxDelta2 = idxDelta1 + 1;
    Index idxPhi1 = std::floor(groupAngle / this->dAngleLookup);
    Index idxPhi2 = idxPhi1 + 1;

    // relative distances from indices to position
    double dEps = (epsilon - (idxEps1 * this->dEpsLookup + this->minEpsLookup)) / this->dEpsLookup;
    double dDelta = (delta - (idxDelta1 * this->dDeltaLookup + this->minDeltaLookup)) / this->dDeltaLookup;
    double dPhi = (groupAngle - (idxPhi1 * this->dAngleLookup)) / this->dAngleLookup;

    // trilinear interpolation coefficients
    double p000 = (1. - dEps) * (1. - dDelta) * (1. - dPhi);
    double p001 = (1. - dEps) * (1. - dDelta) * dPhi;
    double p010 = (1. - dEps) * dDelta * (1. - dPhi);
    double p011 = (1. - dEps) * dDelta * dPhi;
    double p100 = dEps * (1. - dDelta) * (1. - dPhi);
    double p101 = dEps * (1. - dDelta) * dPhi;
    double p110 = dEps * dDelta * (1. - dPhi);
    double p111 = dEps * dDelta * dPhi;

    // corresponding values in the lookup table
    Index memoryOffsetEps = this->stepsAngleLookup * this->stepsDeltaLookup;
    double c000 = this->anisotropyScalarLookup[idxEps1 * memoryOffsetEps + idxDelta1 * this->stepsAngleLookup + idxPhi1];
    double c001 = this->anisotropyScalarLookup[idxEps1 * memoryOffsetEps + idxDelta1 * this->stepsAngleLookup + idxPhi2];
    double c010 = this->anisotropyScalarLookup[idxEps1 * memoryOffsetEps + idxDelta2 * this->stepsAngleLookup + idxPhi1];
    double c011 = this->anisotropyScalarLookup[idxEps1 * memoryOffsetEps + idxDelta2 * this->stepsAngleLookup + idxPhi2];
    double c100 = this->anisotropyScalarLookup[idxEps2 * memoryOffsetEps + idxDelta1 * this->stepsAngleLookup + idxPhi1];
    double c101 = this->anisotropyScalarLookup[idxEps2 * memoryOffsetEps + idxDelta1 * this->stepsAngleLookup + idxPhi2];
    double c110 = this->anisotropyScalarLookup[idxEps2 * memoryOffsetEps + idxDelta2 * this->stepsAngleLookup + idxPhi1];
    double c111 = this->anisotropyScalarLookup[idxEps2 * memoryOffsetEps + idxDelta2 * this->stepsAngleLookup + idxPhi2];

    double interpolated = p000 * c000 + p001 * c001 + p010 * c010 + p011 * c011 + p100 * c100 + p101 * c101 + p110 * c110 + p111 * c111;

    return interpolated;
}

RVector TravelTimeDijkstraModellingTTI::paramsToCombinedModel(const RVector & velP, const RVector & epsilon,
        const RVector & delta, const RVector & incl, const RVector & azim) {
    size_t paramSize = velP.size();

    RVector model = RVector(paramSize * 5);

    // copy data (only works without bound check in subscription operator see Vector::copy_)
    std::copy(&velP[0], &velP[paramSize], &model[0]);
    std::copy(&epsilon[0], &epsilon[paramSize], &model[paramSize]);
    std::copy(&delta[0], &delta[paramSize], &model[2 * paramSize]);
    std::copy(&incl[0], &incl[paramSize], &model[3 * paramSize]);
    std::copy(&azim[0], &azim[paramSize], &model[4 * paramSize]);

    return model;
}

} // namespace GIMLI{
